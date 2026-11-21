/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * cli.h — small "--key value" parser for the benchmark / check programs.
 * Unknown keys are rejected against the list of known options, and all parsed
 * options can be dumped into the result JSON "config" object.
 */
#ifndef PDL_CLI_H
#define PDL_CLI_H

#include "pdl/json.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    const char *def;  /* default value as text */
    const char *help;
} pdl_cli_opt;

typedef struct {
    const pdl_cli_opt *opts;
    int nopts;
    const char **values; /* resolved value per option */
    int help;            /* --help was given */
} pdl_cli;

/* Returns 0 on success, -1 on error (message printed when `verbose`). */
int pdl_cli_parse(pdl_cli *cli, const pdl_cli_opt *opts, int nopts, int argc, char **argv, int verbose);
void pdl_cli_free(pdl_cli *cli);
void pdl_cli_print_help(const pdl_cli *cli, const char *prog, const char *summary);
const char *pdl_cli_str(const pdl_cli *cli, const char *name);
long long pdl_cli_int(const pdl_cli *cli, const char *name);
double pdl_cli_dbl(const pdl_cli *cli, const char *name);
/* Splits a comma separated list; returns count, *items newly allocated (free with pdl_str_free_parts + free). */
int pdl_cli_list(const pdl_cli *cli, const char *name, char ***items);
/* Parses a comma separated list of integers into out (max n); returns count. */
int pdl_cli_int_list(const pdl_cli *cli, const char *name, long long *out, int n);
void pdl_cli_write_json(const pdl_cli *cli, pdl_json *j);

#ifdef __cplusplus
}
#endif

#endif /* PDL_CLI_H */
