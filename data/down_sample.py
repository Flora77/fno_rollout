from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat


def downsample_mat_file(input_path, output_path, stride=4):
    """
    对单个 .mat 文件按时间维度降采样：
    每 stride 个时间点保留 1 个。
    """

    mat = loadmat(input_path)

    # 去掉 scipy 自动生成的元信息
    data = {
        k: v for k, v in mat.items()
        if not k.startswith("__")
    }

    # 自动识别时间长度
    if "time" not in data:
        raise KeyError(f"{input_path.name} 中没有 time 变量")

    time_arr = data["time"]
    Nt = time_arr.size

    new_data = {}

    for name, arr in data.items():

        # 只处理 numpy 数组
        if not isinstance(arr, np.ndarray):
            new_data[name] = arr
            continue

        # 处理 time
        if name == "time":
            if arr.ndim == 2:
                # 常见情况：1 x Nt 或 Nt x 1
                if arr.shape[1] == Nt:
                    new_data[name] = arr[:, ::stride]
                elif arr.shape[0] == Nt:
                    new_data[name] = arr[::stride, :]
                else:
                    new_data[name] = arr
            else:
                new_data[name] = arr[::stride]
            continue

        # 处理 eta、phis 等第一维是时间维度的变量
        if arr.ndim >= 1 and arr.shape[0] == Nt:
            new_data[name] = arr[::stride, ...]
        else:
            # x, y, X, Y 等空间变量保持不变
            new_data[name] = arr

    output_path.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        output_path,
        new_data,
        do_compression=True
    )


def batch_downsample_mat(
    input_dir="collected_JONS_bimodal_mat",
    output_dir="collected_JONS_bimodal_mat_1s",
    stride=4
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    mat_files = sorted(input_dir.glob("*.mat"))

    if len(mat_files) == 0:
        raise FileNotFoundError(f"没有在 {input_dir} 中找到 .mat 文件")

    print(f"找到 {len(mat_files)} 个 mat 文件")

    for i, mat_path in enumerate(mat_files, start=1):
        out_path = output_dir / mat_path.name

        downsample_mat_file(
            input_path=mat_path,
            output_path=out_path,
            stride=stride
        )

        print(f"[{i}/{len(mat_files)}] saved: {out_path}")

    print("全部处理完成")


if __name__ == "__main__":
    batch_downsample_mat(
        input_dir="./data/collected_JONS_bimodal_mat",
        output_dir="./data/collected_JONS_bimodal_mat_1s",
        stride=4
    )