# 单星系 DESI → EBGS notebook

打开 **`single_galaxy_demo.ipynb`**，输入 RA、Dec，依次运行所有单元。工具会下载 DESI 裁图，自动接入 CPU 推理，生成 EBGS / synthetic Euclid VIS 图像并显示对照。默认坐标对应示例星系 `1-10166`，也可以切换为本地 FITS 输入。

输入为 DESI `r,z` 或 `g,r,z` 背景扣除的线性通量 FITS。用户可以提供坐标或本地图像，接口不接受、读取或输出误差图。按现有生产流程重采样后取中心 **128×128 像素、12.8″×12.8″**，随后运行 I2SB；不进行分块或拼接。输出采样为 0.1″/pixel，这不是对模型实际分辨率的测量。大星系可能超出该中心视场；生成结果不是实际 Euclid 观测。

## 文件夹内容

| 文件 / 目录 | 用途 |
| --- | --- |
| `single_galaxy_demo.ipynb` | 从坐标下载或本地读图到推理、显示、保存的完整示例 |
| `download_cutout.py`, `downloads/` | RA/Dec 下载工具及已校验的裁图缓存 |
| `inference.py` | CPU / CUDA 单星系接口，默认 CPU float32 |
| `sgm/` | 推理所需项目源码，独立于原仓库 |
| `resampling.py`, `display_utils.py` | 原生产重采样函数、默认 `arcsinh_rgb` |
| `configs/model.yaml` | 推理配置，原生 PyTorch attention |
| `configs/production_model.yaml` | 用于追溯的原始模型配置 |
| `weights/best_ema.safetensors` | 完整 float32 EMA 推理权重，约 124 MiB |
| `weights/provenance.json` | 原 checkpoint、导出权重的哈希和元数据 |
| `example/BGSUB/` | 同一个星系的 grz 与 rz 两种输入 FITS |
| `requirements.txt`, `install.py` | 第三方 Python 依赖版本和跨平台安装入口 |
| `requirements-tested-lock.txt` | 实测环境的完整 pip 版本清单，含间接依赖 |
| `validate_demo.py`, `validate_download.py`, `validation/` | CPU / 下载验证脚本、原流程参考数据、实测记录 |
| `outputs/` | 已执行示例的生成 FITS、JSON 和对照 PNG；重跑会建新子目录 |
| `source_manifest.json`, `SHA256SUMS` | 源码追溯与交付文件校验值 |

复制整个文件夹即可，不需要原 `DESI2Euclid` 仓库、`/datapool`、PSF 数据或其他模型权重。文件夹包含项目源码依赖、权重和示例数据；第三方 Python 软件包通过安装脚本取得。**首次安装及下载新坐标需要联网；本地文件或已有下载缓存可离线推理。** 不包含预装虚拟环境、Python 解释器或各操作系统的离线 wheel 集合。

## 安装和打开

使用 **Python 3.11–3.13**。以下命令均在本文件夹中执行。

如果通过 GitHub 克隆仓库，请先安装 [Git LFS](https://git-lfs.com/)，再取得完整模型权重：

```bash
git lfs install
git clone https://github.com/Rh-YE/EBGS.git
cd EBGS
git lfs pull
cd single_galaxy_demo
```

GitHub 中的 `weights/best_ema.safetensors` 使用 Git LFS 存储，实际大小为 **129,668,236 字节**。如果该文件只有几行文本，说明取得的是 LFS 指针，需先在仓库内运行 `git lfs pull`。原先交付的完整 ZIP 已含实际权重，无需这一步。

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python install.py
python -m jupyterlab single_galaxy_demo.ipynb
```

Windows PowerShell（无需更改脚本执行策略）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe install.py
.\.venv\Scripts\python.exe -m jupyterlab single_galaxy_demo.ipynb
```

`install.py` 在 Linux / Windows 上安装官方 CPU 版 torch / torchvision，在 macOS 上安装该平台的 PyTorch 包。实际验证的平台和耗时记录在 `validation/cpu_validation.json`；Windows / macOS 未在这台服务器上实测。

Jupyter 中需选择本虚拟环境的 Python 内核。从上述命令启动即可；如在已有 Jupyter/VS Code 中打开，可先注册：

```bash
python -m ipykernel install --user --name desi2euclid-demo --display-name "DESI2Euclid demo"
```

## CPU 和 GPU

Notebook 默认 `DEVICE = "cpu"`、`CPU_THREADS = 4`，使用 float32、原生 attention、单个空间裁剪，不使用 autocast、xformers、torch.compile 或多进程 DataLoader。

默认 `NUM_SAMPLES = 3`、`NFE = 10`、`SPLIT_RATIO = 0.7` 与此前 MaNGA 推理配置一致。3 次采样针对同一中心图像，反变换到物理数值空间后求均值。设置 `NUM_SAMPLES = 1` 可降低计算量，但会改变生成结果。

需要 NVIDIA GPU 时，在单独环境中按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与你的驱动匹配的 torch 2.8.0 / torchvision 0.23.0 CUDA 构建，再执行 `python install.py --keep-torch`，并设置 `DEVICE = "cuda:0"`。`DEVICE = "auto"` 有 CUDA 则使用 CUDA，否则回退 CPU。本示例未启用 Apple MPS；Mac 可使用 CPU。

## 输入 RA、Dec 自动下载并推理

在 notebook 第一个代码单元设置：

```python
INPUT_MODE = "coordinates"
RA = 199.2955574
DEC = 0.913561334
INPUT_BANDS = "grz"  # 或 "rz"
```

RA、Dec 为 ICRS 十进制度数，范围分别为 `[0,360)`、`[-90,90]`。运行下载单元后，`IMAGE_PATH` 会自动指向下载好的 FITS；继续运行后面的单元即可生成 EBGS。

工具使用 [Legacy Surveys 官方 FITS 裁图接口](https://www.legacysurvey.org/dr10/description/#obtaining-images-and-raw-data)，默认 `layer="ls-dr10"`（DR9 北区 / DR10 南区组合）。只请求观测图像 `IMAGE`，下载尺寸固定为 128×128、0.262″/pixel、视场约 33.5″，保留 grz 或 rz 通道及天球 WCS。随后使用原 `prepare_input` 自动准备 0.1″/pixel、中心 12.8″ 的模型输入，不会把 33.5″ 整幅图缩小塞进模型。

官方 coadd 图像已经扣除天空背景，此工具保留线性通量，不额外扣背景或做显示拉伸。Viewer 会按目标 WCS 重采样，因此下载像素与原来直接切 brick 的像素不保证逐位一致。

也可独立调用，无需先加载模型：

```python
from download_cutout import download_cutout
from inference import prepare_input

image_path = download_cutout(199.2955574, 0.913561334, bands="rz")
desi, header, info = prepare_input(image_path)
```

或者在终端下载：

```bash
python download_cutout.py --ra 199.2955574 --dec 0.913561334 --bands grz
```

文件保存到本文件夹的 `downloads/`；旁边的 JSON 记录请求参数、官方 URL、下载时间和 SHA-256。相同参数已有完整且校验通过的缓存时，不重复联网、不覆盖文件。下载先验证格式、完整性、波段、形状、WCS 和数据，再从临时文件发布正式 FITS；HTML、损坏文件或未通过下述基础覆盖检查的响应不会进入推理。

仅使用图像做基础覆盖检查：拒绝 NaN/Inf、常数波段以及任何精确零值，避免把常见的零填充空白区域送入模型。这是保守检查，也可能拒绝合法零值，并非完整曝光覆盖审计。报错时检查坐标与巡天覆盖，不应把缺失像素手工填零后继续生成。

## 使用自己的本地 FITS

设置 `INPUT_MODE = "file"`，然后在第一个代码单元设置 `IMAGE_PATH`、`INPUT_BANDS` 和 `INPUT_PIXEL_SCALE`。输入主 HDU 支持 `(2,H,W)`（顺序 `r,z`）和 `(3,H,W)`（顺序 `g,r,z`），单位和预处理应与训练的 DESI BGSUB 一致；不可使用 RGB/JPEG 或已经拉伸过的数组。

`INPUT_BANDS = "auto"` 优先读取 FITS 的 `BANDS` 字段；没有该字段时，2 通道按 `rz`、3 通道按 `grz` 解释。也可显式设为 `"rz"` 或 `"grz"`。通道数、声明顺序或头信息冲突会报错，避免波段对应错误。

附带 `example/BGSUB/1-10166.fits`（grz）和 `example/BGSUB/1-10166_rz.fits`（rz）。后者是前者 r/z 通道的精确提取，WCS 相同。在文件模式中切换 `IMAGE_PATH`，并设置 `INPUT_BANDS="auto"`，即可尝试两种输入。

保留原生产面积平均重采样、不额外做零点或像素面积通量变换。WCS 存在时会检查像素尺度并传递到输出；该快速路径不支持带畸变的 WCS。输入必须覆盖中心 12.8″，遇到 NaN/Inf 会报错，不会把未覆盖天区伪造为有效输入。

模型以 r 波段作为桥起点、z 波段作为条件；g 波段只用于 DESI 彩色显示。两通道路径严格沿用原 r/z 通道的像素变换参数，r/z 数据和随机种子相同时，两种输入产生相同的 EBGS 图像。用户输入和保存结果中均无需额外通道。

显示 grz 时复用原 `main.py` 的默认 `arcsinh_rgb(..., clip=True)`。rz 则分别显示 r、z 波段的 arcsinh 灰度图，不合成缺失的 g 波段。显示 EBGS 时用 arcsinh 拉伸和完整数据范围，不做百分位截断。模型保留训练时的像素变换；显示设置不改变保存数组。

## 权重和输出

原 checkpoint：`logs/2026-06-01T20-48-48_generation-astroIR_SBn/checkpoints/best.ckpt`，epoch **332**、global step **86580**。

SHA-256：`788dbab951dd7f127aaada1f142191deaa52151708cb69c99da5ea456bd7bcb8`。

导出权重只移除了优化器、训练状态和重复的非 EMA 权重；逐张量确认与原 EMA 值完全相同，没有量化。运行配置 `use_ema=False` 表示 EMA 已折入权重，不需再建立另一份 EMA 缓冲。加载使用 `strict=True` 并校验权重哈希。

每次 notebook 执行会新建 `outputs/<目标>_<设备>_<唯一后缀>/`，保存：

- `*_ebgs.fits`：只有一个主 HDU，二维 float32 生成图像和 WCS，不含输入图或误差图；含 FITS CHECKSUM / DATASUM。
- `*_ebgs.json`：输入、权重、设备、随机种子、采样参数和耗时。
- `*_comparison.png`：相同中心视场的 DESI / EBGS 对照。

## 验证

```bash
python validate_demo.py
python validate_download.py
```

检查真实单星系的预处理与原 Dataset 是否一致、EMA 导出后的预测与原 checkpoint CPU float32 参考结果是否一致、rz/grz 输出逐像素等价、缺省和显式波段识别、无效波段声明拒绝、WCS 映射，以及项目模块是否全部来自本文件夹。参考推理使用原模型源码和原 checkpoint，仅将 attention 后端切换为 CPU 支持的原生实现；参考结果不是 Euclid 真值。

`validate_download.py` 使用实际下载的缓存验证联动，以及 HTML、截断响应、波段/坐标错误、零填充拒绝和缓存复用；不会访问外网。`validation/download_validation.json` 记录检查结果。

`validation/notebook_execution.json` 和 `validation/rz_notebook_execution.json` 分别记录坐标模式 grz/rz 的完整执行；文件模式验证记录在 `validation/file_notebook_execution.json`。CPU 与生产 GPU bf16 路径、不同系统和数学库之间不保证位级一致。运行验证与模型生成细节的科学真实性验证是不同的事项。

2026-10-08 已在独立 Linux 环境、Python 3.13.9、`torch 2.8.0+cpu` 下验证；未安装 xformers、无 CUDA。当前版本的 rz/grz 等价性、运行时间和内存实测记录见 `validation/cpu_validation.json`。Notebook 的两种输入分支均执行验证。服务器实测性能不代表普通笔记本的运行速度。
