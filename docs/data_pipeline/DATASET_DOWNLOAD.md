# 公开数据下载清单

日期：2026-11-02更新。由用户自行下载；助手只做本地离线核查，不执行联网下载。

当前主实验只依赖已经核验并进入团队包的HoliSafe-Bench。JailBreakV-28K保留为可选的外部攻击挑战集候选，不是第三周主包或基础实验的依赖；CIDER继续作为图像扰动专项候选。当前不下载完整JailBreakV、VLGuard或MM-SafetyBench，避免重复下载和在标签、许可未核实前扩大数据范围。

## 1 HoliSafe-Bench

官方文件页：https://huggingface.co/datasets/etri-vilab/holisafe-bench/tree/main

先登录Hugging Face，阅读并接受作者访问条件。下载以下内容：

- 根目录`holisafe_bench.json`
- 完整`images/`目录，保留所有分类和子目录
- 根目录`README.md`

放到：`public_sources\holisafe\`。

```text
public_sources/holisafe/
  README.md
  holisafe_bench.json
  images/
    hate/
    illegal_activity/
    privacy/
    self_harm/
    sexual_content/
    specialized_advice/
    violence/
```

上述JSON和图片目录是一套完整的读取方式，不必重复下载`data/`里的五个Parquet分片。作者描述规模为4031个图文输入、1796张图片；实际数量、缺失和重复情况在下载后重新验证。生成文本和生成图片按作者CC BY-NC 4.0声明处理，其他图片保留各自来源许可，不将整个数据包视为可任意公开上传。

## 2 可选：JailBreakV-28K（当前不参与主实验）

官方目录：https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/tree/main/JailBreakV_28K

本机已有CSV元数据和少量图片，已完成离线核查，但不纳入当前主包。完整图片仍需通过官方作者渠道取得；Hugging Face目录不保证附带完整图片。官方GitHub README提供“Download datasets with images”作者表单：https://github.com/SaFo-Lab/JailBreakV_28K 。只有后续需要独立攻击挑战集、并完成图片完整性、许可和目标模型成功标签定义后，才考虑补齐。

```text
public_sources/jailbreakv/
  README.md
  JailBreakV_28K/
    JailBreakV_28K.csv
    mini_JailBreakV_28K.csv
    RedTeam_2K.csv
    figstep/
    llm_transfer_attack/
    query_related/
```

不能只下载CSV，必须有image_path指向的实际图片。本机已核对：28,000行中仅360行有图，27,640行缺图；mini是full的精确子集，不拼接重复记录。下载不等于采用全部样本；如果将来启用，应单独建立攻击挑战集和独立评测协议，不能直接并入HoliSafe的四划分。

## 可选的批量下载命令

HoliSafe和JailBreakV的CSV均已下载，本轮不必重复下载。若后续重新获取HoliSafe图片，可在你自己的PowerShell终端运行其命令；Hugging Face命令只能拿到该仓库可见文件，**不能替代JailBreakV官方README里的完整图片包**。命令由你执行，助手未安装工具或运行下载。

```powershell
python -m pip install --upgrade huggingface_hub
hf auth login

$holisafeArgs = @(
    'download', 'etri-vilab/holisafe-bench',
    '--repo-type', 'dataset',
    '--include', 'README.md',
    '--include', 'holisafe_bench.json',
    '--include', 'images/**',
    '--local-dir', 'public_sources\holisafe'
)
hf @holisafeArgs

$jailbreakArgs = @(
    'download', 'JailbreakV-28K/JailBreakV-28k',
    '--repo-type', 'dataset',
    '--include', 'README.md',
    '--include', 'JailBreakV_28K/**',
    '--local-dir', 'public_sources\jailbreakv'
)
hf @jailbreakArgs
```

本机新版CLI中，每个匹配项须分别带一个`--include`。不要写成`--include "README.md" "images/**"`：第二项会被当成具体文件名，在Windows下可能导致`**.lock`的WinError 123。上述两组参数已用本机CLI离线解析验证，下载函数被替换为测试桩，未进行任何下载。若`hf`不在PATH中，改用`& .\.venv\Scripts\hf.exe @holisafeArgs`或`& .\.venv\Scripts\hf.exe @jailbreakArgs`。

CLI说明：https://huggingface.co/docs/huggingface_hub/guides/cli 。可在`hf download`命令末尾加`--dry-run`先查看下载清单和预计大小。保留`.cache/huggingface`下载元信息以便核对commit版本。若作者限制访问而尚未授权，等待正规授权；不使用镜像绕过。

## 下载后的工作

提供本机保存路径即可，目录结构不同也可适配。随后核对文件完整性、原始图文对应、官方标签含义、跨来源重复、图片家族和请求组，再形成正式数据包。未下载完成、未核实的资源不计入完成数量。
