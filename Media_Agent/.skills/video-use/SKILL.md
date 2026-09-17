---
name: video-use
description: 口播视频后期剪辑的标准流程与硬性约束。当用户要求剪辑口播视频、加字幕、生成动画素材、配背景音乐、渲染 MP4 时加载本技能。涵盖 funasr/百炼转写生成 SRT、HyperFrames 动画素材、上下排布穿插、BGM 混音与最终渲染。
---

# 口播视频剪辑规范（video-use）

## 什么时候用

用户给了一段口播视频，要求做后期：加速、剪冗余、加字幕、加动画素材、配背景音乐、渲染成片。

## 工作目录（最重要的一条，先看这个）

你运行在一个文件沙箱里，**当前工作目录就是沙箱根目录**。

- 所有输出一律用 `os.path.join(os.environ['WORK_DIR'], '文件名')` 构造。
- 写文件前先 `os.makedirs(os.path.dirname(out), exist_ok=True)`。
- **禁止**手写带盘符的路径（`C:\`、`D:\`）。
- **禁止**使用 `/c/Users` 这种 Git Bash 风格路径。
- 源视频和 BGM 的绝对路径会由任务提示给出，直接读取即可，不用复制进沙箱。

## 运行环境（实测结论，别踩）

- `execute` 工具跑的是 **Windows 的 cmd.exe，不是 bash**。
  列目录用 `dir` 不用 `ls`；看文件用 `type` 不用 `cat`；**不要用 pwd / rm**。
  想列目录或读文件，优先用 `ls` / `read_file` 这些**文件工具**（它们跨平台），
  只有真的要跑代码时才用 `execute`。
- 跑 Python 直接用 `python 脚本名.py`，PATH 已经指向虚拟环境（不要写全路径）。
- `npx` 一律加 `--yes`；**不要**执行 `playwright install`。

## 选型（硬性要求）

**所有视频处理必须用 `moviepy` 库。**

- **禁止**在脚本里写 `subprocess.run(['ffmpeg', ...])` 或 `ffprobe`。
  moviepy 内部会自己调 ffmpeg，你只需要用它提供的 Python API。
- **禁止**用 `ffprobe` 探测元数据 —— 用 `clip.duration` / `clip.size` / `clip.fps`。

## moviepy 2.x 常用 API（本机 2.1.2 实测可用）

```python
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip,
    concatenate_videoclips, concatenate_audioclips, TextClip, ColorClip,
    ImageClip, vfx, afx,
)
# ⚠️ SubtitlesClip 不在顶层！必须从子模块导入：
from moviepy.video.tools.subtitles import SubtitlesClip
```

| 需求 | 写法 |
|---|---|
| 加速 | `clip.with_effects([vfx.SpeedX(1.3)])` |
| 裁剪时间段 | `clip.subclipped(start, end)` |
| 缩放 | `clip.resized(new_size)` 或 `clip.resized(height=426)` |
| 改音量 | `clip.with_effects([afx.VolumeX(3.0)])` |
| 拼接 | `concatenate_videoclips([c1, c2, ...])` |
| 定位 | `clip.with_position(('center', y))` |
| 混音 | `CompositeAudioClip([voice, bgm])` |
| 输出 | `clip.write_videofile('out.mp4', codec='libx264', audio_codec='aac')` |
| 取音频 | `clip.audio`（可能为 None，先判空） |
| 去音轨 | `clip.without_audio()` |

`TextClip` 签名（2.x 与 1.x 不同，别照抄旧教程）：

```python
TextClip(font=..., text=..., font_size=..., size=(w, None),
         color='white', stroke_color='black', stroke_width=2,
         method='caption', text_align='center')
```

> ⚠️ 2.x 用 `font`（字体名或字体文件路径），**没有** `fontsize=`（那是 1.x）。
> 中文字体用 `'Microsoft YaHei'` 或直接给 `C:/Windows/Fonts/msyh.ttc`。

## 五条必须严格遵守的约束

违反其中任何一条，成片会出现**字幕重复**或**下半截被裁掉**。

### 1. 字幕只生成一次，且只能用 SubtitlesClip

正确流程：**转写 → 用时间戳生成一个 SRT 文件 → 用 `SubtitlesClip` 加载它，仅此一次**。

```python
from moviepy.video.tools.subtitles import SubtitlesClip

def make_st(txt):
    return TextClip(
        text=txt, font='Microsoft YaHei', font_size=52,
        color='white', stroke_color='black', stroke_width=2,
        size=(w, None), method='caption',
    ).with_position(('center', h * 0.75))   # ← 关键！定位在原视频区域内

subtitles = SubtitlesClip('subtitles.srt', encoding='utf-8', make_textclip=make_st)
```

**严禁**用 `for` 循环手工创建 `TextClip` 逐行加字幕。
**严禁**同时用 `SubtitlesClip` 和手工 `TextClip`。
**必须**传 `make_textclip` 参数并显式 `with_position` ——
`SubtitlesClip` 默认会把字幕定位到**合成帧的最底部**（那是素材区域），
不指定位置的话字幕下半截会被裁掉。

### 2. 禁止隐式蒙版

**严禁**调用 `.with_mask()` / `.set_mask()` / `.to_mask()`。
**严禁**对视频 clip 调 `.resized((w, h))` 强行改尺寸（会引入蒙版）。

素材保持宽高比，禁止拉伸；输出分辨率与源视频完全一致。

### 3. CompositeVideoClip 的 size 必须包含全部层高

```python
# 上半部分原视频 + 下半部分素材（高度为原视频的 1/3）
composite = CompositeVideoClip(
    [top, bottom.with_position((0, top_h))],
    size=(w, top_h + bottom_h),     # ← 必须是两者之和
)
```

写成 `size=(w, h)` 会在 `h` 位置产生**隐式裁剪蒙版**，
把下面的素材层和字幕下半截全部裁掉。

### 4. 字幕位置只能落在原视频区域内

- y 坐标范围：`h * 0.02` ~ `h * 0.65`（`h` 是**原视频**高度，不是合成帧高度）。
- 推荐 `y = h * 0.62 - clip_h // 2`，保证字幕底部离 `h` 边界至少留 20px。
- 字幕 clip 放在 `CompositeVideoClip` 列表的**最后一项**（最上层）。

### 5. 这些约束必须写进你产出的 Python 脚本注释里

不是可选项 —— 脚本要能自解释为什么这么写。

## 标准流程

### 第 1 步：分析源素材

```python
from moviepy import VideoFileClip
clip = VideoFileClip(源视频路径)
print('时长', clip.duration, '尺寸', clip.size, '帧率', clip.fps, '有音轨', clip.audio is not None)
```

### 第 2 步：转写并生成 SRT

用项目自带的转写工具（**不要**自己写 funasr 调用）：

```python
import sys, os
# 项目根已在 sys.path 里（.pth），可直接 import
from tools.audio_transcriber import transcribe_to_srt

srt_path = transcribe_to_srt(音频文件路径, os.path.join(os.environ['WORK_DIR'], 'subtitles.srt'))
if not srt_path:
    print('[警告] 转写失败，本次不加字幕')
```

先用 `ffmpeg` 抽出音频？**不用** —— `transcribe_to_srt` 接受视频路径，
内部会处理。抽音频这一步由它自己完成。

> 转写走的是**百炼 Fun-ASR-Flash 云端接口**（需 `DASHSCOPE_API_KEY`），
> 不是本地模型。失败时 `transcribe_to_srt` 返回空串，此时**跳过字幕继续渲染**，
> 不要因为没字幕就整个任务停下来。

### 第 3 步：生成动画叠加素材

优先用 HyperFrames（HTML/CSS/GSAP → MP4）：

```bash
npx --yes hyperframes render --input 素材.html --output 素材1.mp4
```

**HyperFrames 不可用时的降级方案**（必须实现）：
直接用 moviepy 生成纯色/文字动画 —— `ColorClip` + `TextClip` + `with_position` 做位移动画，
`vfx` 做淡入淡出。**不要因为 HyperFrames 装不上就停下整个任务。**

### 第 4 步：排版穿插

- 开场素材放**最上层**，尺寸与原视频像素一致。
- 其余素材高度为原视频的 **1/3**，放在原视频**下方**。
- 按口播内容节奏在不同时间点穿插。
- 有的上下排布：上面是原口播视频，下面是素材。
- **空素材的地方必须全屏显示原视频，不能黑屏。**
- 不要上下都用原视频。

### 第 5 步：配背景音乐

- 任务提示会给出 BGM 绝对路径；**没给或文件不存在就跳过混音**，不要报错停下。
- BGM 时长短于视频就循环（`afx.AudioLoop` 或 `concatenate_audioclips` 重复拼接）。
- 衔接处**去掉开头空白**（第一段保留）。
- 音量：**BGM 降到 10%，整体放大到 300%**。

```python
from moviepy import CompositeAudioClip, afx
mixed = CompositeAudioClip([
    voice.with_effects([afx.VolumeX(3.0)]),
    bgm.with_effects([afx.VolumeX(0.1)]),
])
composite = composite.with_audio(mixed)
```

### 第 6 步：渲染

```python
out = os.path.join(os.environ['WORK_DIR'], 'mashup_final.mp4')
composite.write_videofile(out, codec='libx264', audio_codec='aac')
print('最终输出:', os.path.abspath(out))
```

**最后一行必须报告输出的绝对路径** —— 上层靠解析这句话拿结果。

## 中文字体（不写就全是方框）

HyperFrames 生成的每个 HTML 文件，`<head>` 里**必须**加：

```html
<style>body{font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif}</style>
```

moviepy 的 `TextClip` 用 `font='Microsoft YaHei'` 或字体文件绝对路径。

不写这条，中文内容渲染出来全是方框，等于白做。

## 自检清单（交付前过一遍）

- [ ] 所有路径都用 `os.environ['WORK_DIR']` 构造，没有盘符
- [ ] `SubtitlesClip` 从 `moviepy.video.tools.subtitles` 导入
- [ ] 字幕只加了一次，用了 `make_textclip` + `with_position`
- [ ] 没有 `.with_mask()` / `set_mask()`
- [ ] `CompositeVideoClip` 的 `size` 是 `(w, top_h + bottom_h)`
- [ ] 中文字体已指定
- [ ] 空素材区间没有被黑屏填充
- [ ] 循环里没有重复加字幕
- [ ] 最后打印了输出文件的**绝对路径**
- [ ] 脚本里用注释写明了上面这些约束的原因
