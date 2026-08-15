#ifndef FOX_PLANNER_INTERNAL_H
#define FOX_PLANNER_INTERNAL_H

#include "fox/fox.h"

struct fox_plan {
    size_t count;
    fox_plan_item *items;
    fox_plan_summary summary;
};

#endif
