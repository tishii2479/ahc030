import abc
import argparse
import dataclasses
import datetime
import json
import logging
import os
import subprocess
from logging import FileHandler, StreamHandler, getLogger
from typing import List, Optional, Type

import pandas as pd
from joblib import Parallel, delayed


@dataclasses.dataclass
class IInput:
    @abc.abstractmethod
    def __init__(self, in_file: str) -> None:
        raise NotImplementedError()


@dataclasses.dataclass
class IResult:
    @abc.abstractmethod
    def __init__(self, stderr: str, input_file: str, solver_version: str) -> None:
        raise NotImplementedError()


class Runner:
    def __init__(
        self,
        input_class: Type[IInput],
        result_class: Type[IResult],
        solver_cmd: str,
        solver_version: str,
        database_csv: str,
        log_file: str,
        input_csv: Optional[str] = None,
        verbose: int = 10,
    ) -> None:
        self.input_class = input_class
        self.result_class = result_class
        self.solver_cmd = solver_cmd
        self.solver_version = solver_version
        self.database_csv = database_csv
        self.input_csv = input_csv
        self.logger = self.setup_logger(log_file=log_file, verbose=verbose)

    def run_case(self, input_file: str, output_file: str) -> IResult:
        cmd = f"{self.solver_cmd} < {input_file} > {output_file}"
        cwd = os.path.dirname(__file__)
        proc = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, cwd=cwd)
        stderr = proc.stderr.decode("utf-8")
        result = self.result_class(stderr, input_file, self.solver_version)
        return result

    def run(
        self,
        cases: list[tuple[str, str]],
        verbose: int = 10,
        ignore: bool = False,
    ) -> pd.DataFrame:
        results = Parallel(n_jobs=-1, verbose=verbose)(
            delayed(self.run_case)(input_file, output_file)
            for input_file, output_file in cases
        )
        df = pd.DataFrame(list(map(lambda x: vars(x), results)))
        if not ignore:
            add_header = not os.path.exists(self.database_csv)
            df.to_csv(self.database_csv, mode="a", index=False, header=add_header)

        return df

    def evaluate_absolute_score(
        self,
        columns: Optional[List[str]] = None,
        eval_items: List[str] = ["score"],
    ) -> pd.DataFrame:
        self.logger.info(f"Evaluating absolute score: [{self.solver_version}]...")
        database_df = pd.read_csv(self.database_csv)
        if self.input_csv is not None:
            input_df = pd.read_csv(self.input_csv)
            database_df = pd.merge(database_df, input_df, how="left", on="input_file")
        score_df = database_df[
            database_df.solver_version == self.solver_version
        ].reset_index(drop=True)

        self.logger.info(f"Raw score mean: {score_df.score.mean()}")
        self.logger.info("Top 10 improvements:")
        self.logger.info(score_df.sort_values(by="score", ascending=False)[:10])
        self.logger.info("Top 10 aggravations:")
        self.logger.info(score_df.sort_values(by="score")[:10])

        if columns is not None:
            assert 1 <= len(columns) <= 2
            for column in columns:
                score_df[f"{column}_cut"] = pd.cut(score_df[column], 4)
            cut_columns = list(map(lambda col: f"{col}_cut", columns))
            if len(columns) == 1:
                self.logger.info(
                    score_df.groupby(f"{columns[0]}_cut")["relative_score"].mean()
                )
            elif len(columns) == 2:
                self.logger.info(
                    score_df[eval_items + cut_columns].pivot_table(
                        index=f"{columns[0]}_cut", columns=f"{columns[1]}_cut"
                    )
                )

        return score_df

    def evaluate_relative_score(
        self,
        benchmark_solver_version: str,
        columns: Optional[List[str]] = None,
        eval_items: List[str] = ["score", "relative_score"],
    ) -> pd.DataFrame:
        self.logger.info(
            f"Comparing {self.solver_version} -> {benchmark_solver_version}"
        )
        database_df = pd.read_csv(self.database_csv)
        if self.input_csv is not None:
            input_df = pd.read_csv(self.input_csv)
            database_df = pd.merge(database_df, input_df, how="left", on="input_file")
        score_df = database_df[
            database_df.solver_version == self.solver_version
        ].reset_index(drop=True)
        benchmark_df = database_df[
            database_df.solver_version == benchmark_solver_version
        ].reset_index(drop=True)

        score_df.loc[:, "relative_score"] = score_df.score / benchmark_df.score

        self.logger.info(f"Raw score mean: {score_df.score.mean()}")
        self.logger.info(f"Relative score mean: {score_df['relative_score'].mean()}")
        self.logger.info(
            f"Relative score median: {score_df['relative_score'].median()}"
        )
        self.logger.info("Top 10 improvements:")
        self.logger.info(
            score_df.sort_values(by="relative_score", ascending=False)[:10]
        )
        self.logger.info("Top 10 aggravations:")
        self.logger.info(score_df.sort_values(by="relative_score")[:10])
        self.logger.info(
            f"Longest duration: {score_df.sort_values(by='duration').iloc[-1]}"
        )
        self.logger.info(f"Fail count: {(score_df['score'] == 1000.).sum()}")

        if columns is not None:
            assert 1 <= len(columns) <= 2
            for column in columns:
                score_df[f"{column}_cut"] = pd.cut(score_df[column], 4)
            cut_columns = list(map(lambda col: f"{col}_cut", columns))
            if len(columns) == 1:
                self.logger.info(
                    score_df.groupby(f"{columns[0]}_cut")["relative_score"].mean()
                )
            elif len(columns) == 2:
                self.logger.info(
                    score_df[eval_items + cut_columns].pivot_table(
                        index=f"{columns[0]}_cut", columns=f"{columns[1]}_cut"
                    )
                )

        return score_df

    def evaluate_overall_relative_score(
        self,
        columns: Optional[List[str]] = None,
        eval_items: List[str] = ["score", "relative_score"],
        ignore_solver_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        self.logger.info(f"Evaluate: {self.solver_version}")
        database_df = pd.read_csv(self.database_csv)
        if ignore_solver_prefix is not None:
            database_df = database_df[
                (~database_df.solver_version.str.startswith(ignore_solver_prefix))
                | (database_df.solver_version == self.solver_version)
            ]
        if self.input_csv is not None:
            input_df = pd.read_csv(self.input_csv)
            database_df = pd.merge(database_df, input_df, how="left", on="input_file")
        score_df = database_df[
            database_df.solver_version == self.solver_version
        ].reset_index(drop=True)
        best_scores = (
            database_df.groupby("input_file")["score"].min().rename("best_score")
        )
        score_df = pd.merge(score_df, best_scores, on="input_file", how="left")
        score_df["relative_score"] = score_df["best_score"] / score_df["score"]

        self.logger.info(f"Raw score mean: {score_df.score.mean()}")
        self.logger.info(f"Relative score mean: {score_df['relative_score'].mean()}")
        self.logger.info(
            f"Relative score median: {score_df['relative_score'].median()}"
        )
        self.logger.info("Top 10 improvements:")
        self.logger.info(
            score_df.sort_values(by="relative_score", ascending=False)[:10]
        )
        self.logger.info("Top 10 aggravations:")
        self.logger.info(score_df.sort_values(by="relative_score")[:10])
        self.logger.info(f"Fail count: {(score_df['score'] == 1000.).sum()}")

        if columns is not None:
            assert 1 <= len(columns) <= 2
            for column in columns:
                score_df[f"{column}_cut"] = pd.cut(score_df[column], 4)
            cut_columns = list(map(lambda col: f"{col}_cut", columns))
            if len(columns) == 1:
                self.logger.info(
                    score_df.groupby(f"{columns[0]}_cut")["relative_score"].mean()
                )
            elif len(columns) == 2:
                self.logger.info(
                    score_df[eval_items + cut_columns].pivot_table(
                        index=f"{columns[0]}_cut", columns=f"{columns[1]}_cut"
                    )
                )

        return score_df

    def list_solvers(self) -> pd.DataFrame:
        database_df = pd.read_csv(self.database_csv)
        best_scores = (
            database_df.groupby("input_file")["score"].min().rename("best_score")
        )
        # database_df = database_df[~database_df.solver_version.str.endswith("-1000")]
        # database_df = database_df[~database_df.solver_version.str.endswith("-500")]
        database_df = pd.merge(database_df, best_scores, on="input_file", how="left")
        database_df = database_df[database_df.solver_version.str.endswith("-1000")]
        database_df["relative_score"] = database_df["best_score"] / database_df["score"]
        score_df = (
            database_df[
                ~database_df.solver_version.str.startswith("optuna")
                & ~database_df.solver_version.str.startswith("solver-")
            ]
            .groupby("solver_version")[["relative_score", "score"]]
            .mean()
            .sort_values(by="relative_score", ascending=False)
        )
        self.logger.info(score_df[:50])
        self.logger.info(score_df[-3:])

        return database_df

    def setup_logger(self, log_file: str, verbose: int) -> logging.Logger:
        logger = getLogger(__name__)
        logger.setLevel(logging.INFO)

        if verbose > 0:
            stream_handler = StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            logger.addHandler(stream_handler)

        file_handler = FileHandler(log_file, "a")
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        return logger


@dataclasses.dataclass
class Input(IInput):
    def __init__(self, in_file: str) -> None:
        raise NotImplementedError()


@dataclasses.dataclass
class Result(IResult):
    input_file: str
    solver_version: str
    score: int
    duration: float
    query_count: int

    def __init__(self, stderr: str, input_file: str, solver_version: str):
        self.input_file = input_file
        self.solver_version = solver_version

        json_start = stderr.find("result:") + len("result:")
        result_str = stderr[json_start:]
        try:
            result_json = json.loads(result_str)
        except json.JSONDecodeError as e:
            print(e)
            print(f"failed to parse result_str: {result_str}, input_file: {input_file}")
            exit(1)
        self.score = result_json["score"]
        self.duration = result_json["duration"]
        self.query_count = result_json["query_count"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, default="tools")
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("-l", "--list-solver", action="store_true")
    parser.add_argument("-i", "--ignore", action="store_true")
    parser.add_argument("-n", "--case_num", type=int, default=100)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("-v", "--verbose", type=int, default=10)
    parser.add_argument(
        "-s",
        "--solver-path",
        type=str,
        default="./tools/target/release/tester ./target/release/ahc030",
    )
    parser.add_argument(
        "-a",
        "--solver-version",
        type=str,
        default=f"solver-{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
    )
    parser.add_argument("-b", "--benchmark-solver-version", type=str, default=None)
    parser.add_argument("--database-csv", type=str, default="log/database.csv")
    parser.add_argument("--input-csv", type=str, default="log/input.csv")
    parser.add_argument("--log-file", type=str, default="log/a.log")
    args = parser.parse_args()

    runner = Runner(
        Input,
        Result,
        solver_cmd=args.solver_path,
        solver_version=args.solver_version,
        database_csv=args.database_csv,
        input_csv=args.input_csv,
        log_file=args.log_file,
    )
    columns = ["m", "eps"]

    if args.list_solver:
        runner.list_solvers()
    elif args.eval:
        # runner.evaluate_relative_score(
        #     benchmark_solver_version=args.benchmark_solver_version, columns=columns
        # )
        runner.evaluate_overall_relative_score(columns=columns)
    else:
        subprocess.run("cargo build --features local --release", shell=True)
        subprocess.run(
            f"python3 expander.py > log/backup/{args.solver_version}.rs", shell=True
        )
        cases = [
            (f"{args.data_dir}/in/{seed:04}.txt", f"{args.data_dir}/out/{seed:04}.txt")
            for seed in range(args.start_case, args.start_case + args.case_num)
        ]
        runner.run(cases=cases, ignore=args.ignore, verbose=args.verbose)
        # runner.evaluate_relative_score(
        #     benchmark_solver_version=args.benchmark_solver_version, columns=columns
        # )
        runner.evaluate_overall_relative_score(columns=columns)
