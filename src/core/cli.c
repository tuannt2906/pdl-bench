/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * cli.c — "--key value" option parsing for the auxiliary programs.
 */
#include "pdl/cli.h"

#include <string.h>

#include "pdl/common.h"

static int find(const pdl_cli *cli, const char *name) {
    for (int i = 0; i < cli->nopts; ++i)
        if (strcmp(cli->opts[i].name, name) == 0) return i;
    return -1;
}

int pdl_cli_parse(pdl_cli *cli, const pdl_cli_opt *opts, int nopts, int argc, char **argv, int verbose) {
    cli->opts = opts;
    cli->nopts = nopts;
    cli->help = 0;
    cli->values = (const char **)pdl_xcalloc((size_t)nopts, sizeof(char *));
    for (int i = 0; i < nopts; ++i) cli->values[i] = opts[i].def;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            cli->help = 1;
            continue;
        }
        if (strncmp(a, "--", 2) != 0) {
            if (verbose) fprintf(stderr, "unexpected argument '%s'\n", a);
            return -1;
        }
        char name[64];
        const char *val = NULL;
        const char *eq = strchr(a + 2, '=');
        if (eq) {
            snprintf(name, sizeof(name), "%.*s", (int)(eq - a - 2), a + 2);
            val = eq + 1;
        } else {
            snprintf(name, sizeof(name), "%s", a + 2);
        }
        int k = find(cli, name);
        if (k < 0) {
            if (verbose) fprintf(stderr, "unknown option --%s (see --help)\n", name);
            return -1;
        }
        if (!val) val = (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) ? argv[++i] : "1";
        cli->values[k] = val;
    }
    return 0;
}

void pdl_cli_free(pdl_cli *cli) {
    free((void *)cli->values);
    cli->values = NULL;
}

void pdl_cli_print_help(const pdl_cli *cli, const char *prog, const char *summary) {
    printf("Usage: [mpirun -np K] %s [--option value]...\n\n%s\n\n", prog, summary);
    for (int i = 0; i < cli->nopts; ++i)
        printf("  --%-18s %s [default: %s]\n", cli->opts[i].name, cli->opts[i].help,
               cli->opts[i].def[0] ? cli->opts[i].def : "\"\"");
    printf("\nSee docs/CLI.md for details.\n");
}

const char *pdl_cli_str(const pdl_cli *cli, const char *name) {
    int k = find(cli, name);
    PDL_CHECK(k >= 0, "internal: undeclared option --%s", name);
    return cli->values[k];
}

long long pdl_cli_int(const pdl_cli *cli, const char *name) {
    const char *v = pdl_cli_str(cli, name);
    char *end = NULL;
    long long x = strtoll(v, &end, 10);
    PDL_CHECK(end && *end == '\0', "option --%s expects an integer (got '%s')", name, v);
    return x;
}

double pdl_cli_dbl(const pdl_cli *cli, const char *name) {
    const char *v = pdl_cli_str(cli, name);
    char *end = NULL;
    double x = strtod(v, &end);
    PDL_CHECK(end && *end == '\0', "option --%s expects a number (got '%s')", name, v);
    return x;
}

int pdl_cli_list(const pdl_cli *cli, const char *name, char ***items) {
    const char *v = pdl_cli_str(cli, name);
    int cap = 1;
    for (const char *p = v; *p; ++p) cap += (*p == ',');
    *items = (char **)pdl_xcalloc((size_t)cap, sizeof(char *));
    if (!*v) return 0;
    return pdl_str_split(v, ',', *items, cap);
}

int pdl_cli_int_list(const pdl_cli *cli, const char *name, long long *out, int n) {
    char **items;
    int cnt = pdl_cli_list(cli, name, &items);
    int k = 0;
    for (int i = 0; i < cnt && k < n; ++i) {
        char *end = NULL;
        long long x = strtoll(items[i], &end, 10);
        PDL_CHECK(end && *end == '\0', "option --%s: '%s' is not an integer", name, items[i]);
        out[k++] = x;
    }
    pdl_str_free_parts(items, cnt);
    free(items);
    return k;
}

void pdl_cli_write_json(const pdl_cli *cli, pdl_json *j) {
    pdl_json_begin_object(j, "config");
    for (int i = 0; i < cli->nopts; ++i) {
        char key[64];
        snprintf(key, sizeof(key), "%s", cli->opts[i].name);
        for (char *p = key; *p; ++p)
            if (*p == '-') *p = '_';
        pdl_json_kv_str(j, key, cli->values[i]);
    }
    pdl_json_end_object(j);
}
