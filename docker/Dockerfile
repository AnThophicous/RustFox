FROM alpine:3.20 AS build

RUN apk add --no-cache build-base cmake git

WORKDIR /src
COPY . .

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DFOX_STATIC=ON \
        -DFOX_BUILD_TESTS=OFF \
 && cmake --build build -j "$(nproc)" \
 && strip build/fox

FROM alpine:3.20

LABEL org.opencontainers.image.title="rustfox" \
      org.opencontainers.image.description="Tiered inference core for weak hardware" \
      org.opencontainers.image.source="https://github.com/AnThophicous/RustFox" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN adduser -D -u 10001 fox \
 && mkdir -p /var/cache/rustfox /models \
 && chown -R fox:fox /var/cache/rustfox /models

COPY --from=build /src/build/fox /usr/local/bin/fox

USER fox
ENV FOX_CACHE=/var/cache/rustfox
VOLUME ["/models", "/var/cache/rustfox"]

ENTRYPOINT ["/usr/local/bin/fox"]
CMD ["probe"]
