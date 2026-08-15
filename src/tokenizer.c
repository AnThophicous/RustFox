#include "fox/fox.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

struct fox_tokenizer {
    const fox_gguf *model;
    const char **tokens;
    size_t *lengths;
    float *scores;
    size_t vocab_size;
    uint32_t byte_tokens[256];
    uint32_t bos;
    uint32_t eos;
    uint32_t unk;
    int add_bos;
    int add_eos;
};

static size_t metadata_index(const fox_gguf *model, const char *key)
{
    return fox_gguf_find(model, key);
}

static uint32_t metadata_token_id(const fox_gguf *model, const char *key)
{
    size_t index = metadata_index(model, key);
    uint32_t value;
    if (index == SIZE_MAX || fox_gguf_get_u32(model, index, &value) != FOX_OK)
        return FOX_TOKEN_INVALID;
    return value;
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int byte_token(const char *token, uint8_t *value)
{
    int hi, lo;
    if (strlen(token) != 6 || token[0] != '<' || token[1] != '0' ||
        (token[2] != 'x' && token[2] != 'X') || token[5] != '>') return 0;
    hi = hex_digit(token[3]);
    lo = hex_digit(token[4]);
    if (hi < 0 || lo < 0) return 0;
    *value = (uint8_t)((hi << 4) | lo);
    return 1;
}

static void tokenizer_free_arrays(fox_tokenizer *tokenizer)
{
    free(tokenizer->tokens);
    free(tokenizer->lengths);
    free(tokenizer->scores);
    tokenizer->tokens = NULL;
    tokenizer->lengths = NULL;
    tokenizer->scores = NULL;
}

fox_status fox_tokenizer_open(const fox_gguf *model, fox_tokenizer **out)
{
    fox_tokenizer *tokenizer;
    size_t model_index, tokens_index, scores_index, i;
    const char *model_name;
    size_t count;

    if (!model || !out) return FOX_ERR_ARG;
    *out = NULL;
    model_index = metadata_index(model, "tokenizer.ggml.model");
    tokens_index = metadata_index(model, "tokenizer.ggml.tokens");
    if (model_index == SIZE_MAX || tokens_index == SIZE_MAX)
        return FOX_ERR_NOTFOUND;
    if (fox_gguf_get_string(model, model_index, &model_name) != FOX_OK)
        return FOX_ERR_FORMAT;
    if (strcmp(model_name, "llama") != 0 && strcmp(model_name, "sentencepiece") != 0)
        return FOX_ERR_UNSUPPORTED;
    if (fox_gguf_metadata_array_type(model, tokens_index) != FOX_GGUF_STRING)
        return FOX_ERR_FORMAT;
    count = fox_gguf_metadata_array_count(model, tokens_index);
    if (count == 0 || count > UINT32_MAX) return FOX_ERR_FORMAT;

    tokenizer = (fox_tokenizer *)calloc(1, sizeof(*tokenizer));
    if (!tokenizer) return FOX_ERR_NOMEM;
    tokenizer->model = model;
    tokenizer->vocab_size = count;
    tokenizer->bos = metadata_token_id(model, "tokenizer.ggml.bos_token_id");
    tokenizer->eos = metadata_token_id(model, "tokenizer.ggml.eos_token_id");
    tokenizer->unk = metadata_token_id(model, "tokenizer.ggml.unknown_token_id");
    tokenizer->add_bos = tokenizer->bos != FOX_TOKEN_INVALID;
    tokenizer->add_eos = 0;
    model_index = metadata_index(model, "tokenizer.ggml.add_bos_token");
    if (model_index != SIZE_MAX) {
        if (fox_gguf_get_bool(model, model_index, &tokenizer->add_bos) != FOX_OK) {
            fox_tokenizer_close(tokenizer);
            return FOX_ERR_FORMAT;
        }
    }
    model_index = metadata_index(model, "tokenizer.ggml.add_eos_token");
    if (model_index != SIZE_MAX) {
        if (fox_gguf_get_bool(model, model_index, &tokenizer->add_eos) != FOX_OK) {
            fox_tokenizer_close(tokenizer);
            return FOX_ERR_FORMAT;
        }
    }

    tokenizer->tokens = (const char **)calloc(count, sizeof(*tokenizer->tokens));
    tokenizer->lengths = (size_t *)calloc(count, sizeof(*tokenizer->lengths));
    tokenizer->scores = (float *)calloc(count, sizeof(*tokenizer->scores));
    if (!tokenizer->tokens || !tokenizer->lengths || !tokenizer->scores) {
        fox_tokenizer_close(tokenizer);
        return FOX_ERR_NOMEM;
    }
    for (i = 0; i < 256; i++) tokenizer->byte_tokens[i] = FOX_TOKEN_INVALID;
    for (i = 0; i < count; i++) {
        const char *token;
        uint8_t byte;
        if (fox_gguf_get_array_string(model, tokens_index, i, &token) != FOX_OK) {
            fox_tokenizer_close(tokenizer);
            return FOX_ERR_FORMAT;
        }
        tokenizer->tokens[i] = token;
        tokenizer->lengths[i] = strlen(token);
        if (byte_token(token, &byte)) tokenizer->byte_tokens[byte] = (uint32_t)i;
    }
    scores_index = metadata_index(model, "tokenizer.ggml.scores");
    if (scores_index != SIZE_MAX) {
        if (fox_gguf_metadata_array_type(model, scores_index) != FOX_GGUF_FLOAT32 ||
            fox_gguf_metadata_array_count(model, scores_index) != count) {
            fox_tokenizer_close(tokenizer);
            return FOX_ERR_FORMAT;
        }
        for (i = 0; i < count; i++) {
            if (fox_gguf_get_array_f32(model, scores_index, i,
                                       &tokenizer->scores[i]) != FOX_OK) {
                fox_tokenizer_close(tokenizer);
                return FOX_ERR_FORMAT;
            }
        }
    }
    *out = tokenizer;
    return FOX_OK;
}

void fox_tokenizer_close(fox_tokenizer *tokenizer)
{
    if (!tokenizer) return;
    tokenizer_free_arrays(tokenizer);
    free(tokenizer);
}

size_t fox_tokenizer_vocab_size(const fox_tokenizer *tokenizer)
{
    return tokenizer ? tokenizer->vocab_size : 0;
}

uint32_t fox_tokenizer_bos(const fox_tokenizer *tokenizer)
{
    return tokenizer ? tokenizer->bos : FOX_TOKEN_INVALID;
}

uint32_t fox_tokenizer_eos(const fox_tokenizer *tokenizer)
{
    return tokenizer ? tokenizer->eos : FOX_TOKEN_INVALID;
}

static int make_sentencepiece_input(const char *text, char **out, size_t *length)
{
    size_t input_length = strlen(text);
    size_t capacity;
    size_t i, n = 0;
    char *normalized;
    unsigned char *bytes;

    if (input_length > (SIZE_MAX - 4) / 4) return 0;
    capacity = input_length * 4 + 4;
    normalized = (char *)malloc(capacity);
    if (!normalized) return -1;
    bytes = (unsigned char *)normalized;
    bytes[n++] = 0xE2;
    bytes[n++] = 0x96;
    bytes[n++] = 0x81;
    for (i = 0; i < input_length; i++) {
        if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
            bytes[n++] = 0xE2;
            bytes[n++] = 0x96;
            bytes[n++] = 0x81;
        } else {
            bytes[n++] = (unsigned char)text[i];
        }
    }
    normalized[n] = '\0';
    *out = normalized;
    *length = n;
    return 1;
}

fox_status fox_tokenizer_encode(const fox_tokenizer *tokenizer, const char *text,
                                uint32_t *tokens, size_t capacity, size_t *written)
{
    char *normalized;
    size_t normalized_length, i, j, max_tokens, count = 0;
    float *best_score;
    size_t *previous;
    uint32_t *previous_token;
    uint32_t *result;
    int made_input;
    fox_status status = FOX_OK;

    if (!tokenizer || !text || !written || (capacity && !tokens)) return FOX_ERR_ARG;
    *written = 0;
    made_input = make_sentencepiece_input(text, &normalized, &normalized_length);
    if (made_input == 0) return FOX_ERR_FORMAT;
    if (made_input < 0) return FOX_ERR_NOMEM;
    if (normalized_length > (SIZE_MAX - 2) / sizeof(*best_score)) {
        free(normalized);
        return FOX_ERR_NOMEM;
    }
    max_tokens = normalized_length + 2;
    best_score = (float *)malloc((normalized_length + 1) * sizeof(*best_score));
    previous = (size_t *)malloc((normalized_length + 1) * sizeof(*previous));
    previous_token = (uint32_t *)malloc((normalized_length + 1) * sizeof(*previous_token));
    result = (uint32_t *)malloc(max_tokens * sizeof(*result));
    if (!best_score || !previous || !previous_token || !result) {
        free(normalized); free(best_score); free(previous); free(previous_token); free(result);
        return FOX_ERR_NOMEM;
    }
    for (i = 0; i <= normalized_length; i++) {
        best_score[i] = -FLT_MAX;
        previous[i] = SIZE_MAX;
        previous_token[i] = FOX_TOKEN_INVALID;
    }
    best_score[0] = 0.0f;
    for (i = 0; i < normalized_length; i++) {
        if (best_score[i] == -FLT_MAX) continue;
        for (j = 0; j < tokenizer->vocab_size; j++) {
            size_t next = i + tokenizer->lengths[j];
            float candidate;
            if (tokenizer->lengths[j] == 0 || next > normalized_length ||
                memcmp(normalized + i, tokenizer->tokens[j], tokenizer->lengths[j]) != 0)
                continue;
            candidate = best_score[i] + tokenizer->scores[j];
            if (candidate > best_score[next] ||
                (candidate == best_score[next] && (previous_token[next] == FOX_TOKEN_INVALID ||
                 j < previous_token[next]))) {
                best_score[next] = candidate;
                previous[next] = i;
                previous_token[next] = (uint32_t)j;
            }
        }
        if (previous[i + 1] == SIZE_MAX) {
            uint8_t byte = (uint8_t)normalized[i];
            if (tokenizer->byte_tokens[byte] != FOX_TOKEN_INVALID) {
                best_score[i + 1] = best_score[i] - 100.0f;
                previous[i + 1] = i;
                previous_token[i + 1] = tokenizer->byte_tokens[byte];
            } else if (tokenizer->unk != FOX_TOKEN_INVALID) {
                best_score[i + 1] = best_score[i] - 100.0f;
                previous[i + 1] = i;
                previous_token[i + 1] = tokenizer->unk;
            }
        }
    }
    if (previous[normalized_length] == SIZE_MAX) {
        status = FOX_ERR_FORMAT;
        goto done;
    }
    i = normalized_length;
    while (i != 0) {
        if (count == max_tokens) { status = FOX_ERR_INTERNAL; goto done; }
        result[count++] = previous_token[i];
        i = previous[i];
    }
    for (i = 0; i < count / 2; i++) {
        uint32_t temp = result[i];
        result[i] = result[count - i - 1];
        result[count - i - 1] = temp;
    }
    if (tokenizer->add_bos && tokenizer->bos != FOX_TOKEN_INVALID) {
        if (count == max_tokens) { status = FOX_ERR_INTERNAL; goto done; }
        memmove(result + 1, result, count * sizeof(*result));
        result[0] = tokenizer->bos;
        count++;
    }
    if (tokenizer->add_eos && tokenizer->eos != FOX_TOKEN_INVALID) {
        if (count == max_tokens) { status = FOX_ERR_INTERNAL; goto done; }
        result[count++] = tokenizer->eos;
    }
    *written = count;
    if (capacity < count) { status = FOX_ERR_ARG; goto done; }
    if (count) memcpy(tokens, result, count * sizeof(*tokens));

done:
    free(normalized); free(best_score); free(previous); free(previous_token); free(result);
    return status;
}

static int append_char(char *text, size_t capacity, size_t *length, char value)
{
    if (*length + 1 < capacity) text[*length] = value;
    (*length)++;
    return 1;
}

fox_status fox_tokenizer_decode(const fox_tokenizer *tokenizer,
                                const uint32_t *tokens, size_t count,
                                char *text, size_t capacity, size_t *written)
{
    size_t i, j, length = 0;
    int at_sequence_start;
    if (!tokenizer || (!tokens && count) || !written || (capacity && !text))
        return FOX_ERR_ARG;
    at_sequence_start = (count > 0 && tokens[0] == tokenizer->bos);
    for (i = 0; i < count; i++) {
        const char *token;
        uint8_t byte;
        if (tokens[i] >= tokenizer->vocab_size) return FOX_ERR_NOTFOUND;
        if (tokens[i] == tokenizer->bos || tokens[i] == tokenizer->eos) continue;
        token = tokenizer->tokens[tokens[i]];
        if (byte_token(token, &byte)) {
            append_char(text, capacity, &length, (char)byte);
            continue;
        }
        for (j = 0; j < tokenizer->lengths[tokens[i]]; j++) {
            if (j + 2 < tokenizer->lengths[tokens[i]] &&
                (unsigned char)token[j] == 0xe2 &&
                (unsigned char)token[j + 1] == 0x96 &&
                (unsigned char)token[j + 2] == 0x81) {
                if (length != 0 || !at_sequence_start)
                    append_char(text, capacity, &length, ' ');
                j += 2;
            } else {
                append_char(text, capacity, &length, token[j]);
            }
        }
    }
    *written = length;
    if (capacity <= length) return FOX_ERR_ARG;
    text[length] = '\0';
    return FOX_OK;
}
