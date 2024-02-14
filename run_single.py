import ast
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    seed = int(sys.argv[1])
    file = f"{seed:04}"

    subprocess.run("cargo build --features local --release", shell=True)
    subprocess.run(
        "./tools/target/release/tester ./target/release/ahc030"
        + f"< tools/in/{file}.txt > tools/out/{file}.txt",
        shell=True,
    )
    # subprocess.run(
    #     "./tools/target/release/vis" + f" tools/in/{file}.txt tools/out/{file}.txt",
    #     shell=True,
    # )
    subprocess.run(f"pbcopy < tools/out/{file}.txt", shell=True)

    # 過去ログとの比較
    input_df = pd.read_csv("./log/input.csv")
    print(input_df[(input_df.input_file == f"tools/in/{file}.txt")])
    df = pd.read_csv("./log/database.csv")
    print(
        df[(df.input_file == f"tools/in/{file}.txt")][
            ["solver_version", "score"]
        ].sort_values(by="score")[:20]
    )

    with open("./score.log", "r") as f:
        score_log = ast.literal_eval(f.read())

    plt.plot(score_log)
    plt.show()
