# # # src/split.py
# # import sys
# # from pyspark.sql import SparkSession
# # import yaml


# # def split_data(
# #     input_parquet: str, out_train: str, out_val: str, seed: int = 42, frac: float = 0.8
# # ):
# #     spark = SparkSession.builder.appName("TitanicSplit").getOrCreate()
# #     df = spark.read.parquet(input_parquet)
# #     train_df, val_df = df.randomSplit([frac, 1.0 - frac], seed=seed)
# #     train_df.write.mode("overwrite").parquet(out_train)
# #     val_df.write.mode("overwrite").parquet(out_val)
# #     print(
# #         f"Saved train: {out_train} ({train_df.count()} rows), val: {out_val} ({val_df.count()} rows)"
# #     )
# #     spark.stop()



# # # In your split.py file
# # if __name__ == "__main__":
# #     if len(sys.argv) < 4:
# #         # This part of your code seems to be correct
# #         print("Usage: python src/split.py <input_parquet> <out_train_parquet> <out_val_parquet> --params <params_path>")
# #         sys.exit(1)

# #     input_path = sys.argv[1]
# #     out_train = sys.argv[2]
# #     out_val = sys.argv[3]

# #     # Check if a --params argument exists
# #     params_path = "params.yaml"
# #     if "--params" in sys.argv:
# #         try:
# #             params_path = sys.argv[sys.argv.index("--params") + 1]
# #         except IndexError:
# #             print("Error: Missing path for --params")
# #             sys.exit(1)

# #     # Now, pass the arguments to your function
# #     with open(params_path, "r") as f:
# #         params = yaml.safe_load(f)
    
# #     split_data(input_path, out_train, out_val, params["split"]["seed"], params["split"]["train_size"])


# # src/split.py
# import sys
# from pyspark.sql import SparkSession


# def split_data(input_parquet: str, out_train: str, out_val: str, seed: int = 42, frac: float = 0.8):
#     spark = SparkSession.builder.appName("TitanicSplit").getOrCreate()
#     df = spark.read.parquet(input_parquet)
#     train_df, val_df = df.randomSplit([frac, 1.0 - frac], seed=seed)
#     train_df.write.mode("overwrite").parquet(out_train)
#     val_df.write.mode("overwrite").parquet(out_val)
#     print(
#         f"Saved train: {out_train} ({train_df.count()} rows), val: {out_val} ({val_df.count()} rows)"
#     )
#     spark.stop()


# if __name__ == "__main__":
#     if len(sys.argv) < 4:
#         print(
#             "Usage: python src/split.py <input_parquet> <out_train_parquet> <out_val_parquet> [--params <params_path>]"
#         )
#         sys.exit(1)

#     input_path = sys.argv[1]
#     out_train = sys.argv[2]
#     out_val = sys.argv[3]

#     split_data(input_path, out_train, out_val)


import sys
from pyspark.sql import SparkSession

def split_data(input_parquet, out_train, out_val, seed=42, frac=0.8):
    spark = SparkSession.builder.appName("TitanicSplit").getOrCreate()
    df = spark.read.parquet(input_parquet)

    train_df, val_df = df.randomSplit([frac, 1 - frac], seed=seed)

    train_df.write.mode("overwrite").parquet(out_train)
    val_df.write.mode("overwrite").parquet(out_val)

    print(f"Saved train: {out_train} ({train_df.count()} rows), val: {out_val} ({val_df.count()} rows)")
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/split.py <input_parquet> <out_train_parquet> <out_val_parquet> --params <params_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    out_train = sys.argv[2]
    out_val = sys.argv[3]

    split_data(input_path, out_train, out_val)
