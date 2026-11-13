# SpeakerVid-5M: A Large-Scale High-Quality Dataset for Audio-Visual Dyadic Interactive Human Generation

SpeakerVid-5M is the first large-scale, high-quality dataset designed for **audio-visual dyadic interactive virtual human generation**. With over **8,743 hours** of data, it contains more than **5.2 million video clips** of human portraits. The dataset covers a wide variety of interaction types, including monadic talking, listening, and dyadic conversations.

We have open-sourced the data annotation, data collection process, and data usage workflow.
The data annotations can be obtained from [Hugging Face](https://huggingface.co/datasets/dorni/SpeakerVid-5M-Dataset/).

## Official Code and Setup

To set up the environment for the SpeakerVid-5M dataset, follow the steps below:

```bash
conda create -n speaker python=3.8
conda activate speaker
pip install -r requirements.txt
```


## Data Usage ([download from hugging face](https://huggingface.co/datasets/dorni/SpeakerVid-5M-Dataset/))
We provide separate list files for all data and SFT data. The ```all_data_list.json``` file contains the YouTube video IDs and the names of several clips obtained from the video segmentation (these names serve as unique identifiers and can be used to locate the corresponding annotations in the annotation folder). Every YouTube video ID specific to a single video on youtube.com, for example, you can access ```8Hg_-5aUOYo``` through Link ```https://www.youtube.com/watch?v=8Hg_-5aUOYo```. Download the original video and then process it according to our annotations. 
The ```SFT_set.json``` file contains the clip names for the SFT data.

The annotation files in our dataset are organized into several folders, including merge_anno, dwpose, asr, and others. The contents are as follows:

### 1. merge_anno: 
Each annotation file in this folder corresponds to a clip, with filenames following the format: video_id + cur_clip_num + speaker_id + person_id. Each file includes the unique YouTube video ID. After downloading the video, you can determine the time range of the clip based on the ```org_start_seconds``` and ```video_total_duration```, and define the spatial range using the box annotation. Additionally, we provide start and duration options, which allow for a smaller time range to be selected, reducing errors from scene detection. This file also contains our computed clear scores and dover scores, which can be used to filter for high-quality data as needed.

After downloading org videos from youtube, you can use the ```split_clip_from_anno.py``` to split the videos into clips based on ```merge_anno``` annotations. 

### 2. dwpose: 
Due to the large size of the files (1.5TB), we have canceled the transfer of this portion of the data. However, the same data can be computed using the code in  ```cal_skeleton```, with filenames following the same convention as the annotation files. Each .npy file stores the skeletal sequence of the corresponding clip.

### 3. asr: 
This folder includes JSON files, with filenames formatted as video_id + cur_clip_num + speaker_id. Each JSON file contains the ASR (Automatic Speech Recognition) transcription of the audio corresponding to each clip, along with confidence scores and the rate of empty transcriptions. These can be used to filter clips based on the quality of the speech recognition.

### 4. l_score: 
This folder contains clarity scores for the hands and face. We provide both frame-by-frame and average annotations, along with relative and absolute clarity scores.

### 5. anno (caption): 
This folder contains MLLM captions from qwen-vl.
The file includes annotations on whether the subject is moving, speaking, the number of people in the video, the facing direction, full-body/half-body view, the primary subject in the video, the level of physical activity, as well as detailed movement and facial expression captions.

### 6. build dataset for training
The dataset JSON for training is organized in the build_dataset folder, which includes demo code for high-quality data filtering, construction of the two-person conversation dataset, the listening dataset, and the multi-turn conversation dataset. You can customize the dataset construction based on your specific requirements for training. We provide ```traversal_clips.py ``` to obtain ```video_list_txt``` used in building datasets.

### 7. benchmark 
```testset.json```: This file represents the test set containing 500 videos, while SFT_set refers to the high-quality dataset.
The metrics folder contains the code for calculating various evaluation metrics.
The audio folder, along with wavlm_simo_metrics.py, provides the code for evaluating audio quality preservation metrics.
The corh folder includes the code for calculating dialogue consistency metrics.
The express folder, along with expression_metrics.py, contains the code for calculating facial expression FID (Fréchet Inception Distance).
sync_metrics provides the code for evaluating audio-visual synchronization. When using this, ensure that the generated frame rate (FPS) matches the FPS of the ground truth (GT) data.
The vid_metrics folder, along with video_metrics.sh, is used for calculating video quality metrics, inspired by the Disco metric.

### weights
Please download the [asset model](https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models/tree/main/assets)and place it in the metrics/express/assets folder.
Please download the [speaker ckpt](https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models/tree/main/ckpt) and place it in the speaker/ckpt folder.
Please download the [model weights](https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models/tree/main/weights) and place them in the weights folder.



## Data Curation Pipeline

Our data curation pipeline is based on the following works:

- [**3D-Speaker**](https://github.com/modelscope/3D-Speaker)
- [**YOLO**](https://github.com/ultralytics/ultralytics)
- [**DWpose**](https://github.com/IDEA-Research/DWPose)
- [**Whisper**](https://github.com/openai/whisper)
- [**SceneDetect**](https://www.scenedetect.com/)
- [**SyncNet**](https://github.com/joonson/syncnet_python)
- [**ArcFace**](https://github.com/deepinsight/insightface)
- [**Deepface**](https://github.com/serengil/deepface)
- [**UniSpeech**](https://github.com/microsoft/UniSpeech)
- [**Deep3DFaceRecon**](https://github.com/sicxu/Deep3DFaceRecon_pytorch)

We recommend using [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) for downloading videos.

### 1. Base Annotation
This part of the code is located in the cal_base_anno module. It outputs .tar packages to a specified directory, with each .tar containing the corresponding video, audio, and a basic annotation JSON file. The extracted video and audio are synchronized and contain only a single person.

The processing is based on the original videos. We recommend organizing your original videos into multiple batch folders, each containing several JSON files that list video paths. This structure allows for efficient parallel processing.
```
/data
├── batch_01
│   ├── video_list_001.json
│   ├── video_list_002.json
│   └── ...
├── batch_02
│   ├── video_list_001.json
│   ├── video_list_002.json
│   └── ...
├── batch_03
│   ├── video_list_001.json
│   ├── video_list_002.json
│   └── ...
└── ...
```
Example of a single video_list.json
```
[
  "/path/to/your/videos/video1.mp4",
  "/path/to/your/videos/video2.mp4",
]
```

### 2. DWpose Skeleton 
The cal_skeleton folder contains the code for annotating joint sequences using DWpose. This step processes the video clips obtained from the first stage.

You need to collect the paths of all .mp4 files generated in the first step (Base Annotation) into a single .txt file, and pass it using the ```--video_list_path argument```.

### 3. ASR Annotationn 
The cal_asr folder contains the code for generating ASR (Automatic Speech Recognition) results using Whisper. This step processes the audio clips (in .wav format) obtained from the first stage (Base Annotation).
You need to collect the paths of all .wav files generated in the first step into a single .txt file, and pass it using the ```--audio_list_path argument```.

### 4. Blur Score Calculation for Hands and Face
The cal_blur_score folder contains the code for computing the blur scores of hands and face regions. This step depends on the completion of the third step (DWpose joint annotation).
You still need to provide the ```--video_list_path ``` argument, as well as the DWpose output directory via the ```--dwpose_dir``` argument (Step 2 DWpose Skeleton ).

### 5. Luminance Calculation
The cal_luminance folder contains the code for computing video luminance, which is used to filter out clips that are too bright or too dark.
The ```--video_list_path``` argument is still required for this step.

### 6. SceneDetect and Speaker Diarization (Optional)
This part of the code is located in the cal_scene_and_audio_diarization folder. It performs temporal segmentation directly on the original videos, from both visual and audio perspectives.
For the audio part, we use 3D-Speaker to perform speaker diarization, producing the acoustic identity and speaking time segments for each speaker. For visual scene segmentation, we adopt the traditional SceneDetect method.

Note that this step is already integrated into the first stage of processing. You may directly reuse the intermediate results from the first step to skip this one.

## Ethical Statement and Data Usage Policy

The **SpeakerVid-5M** dataset is released to the academic and research communities to foster innovation and advance scientific discovery in the field of digital humans and multimodal interaction. The dataset is made available under the following conditions:

### Purpose of Use

This dataset is strictly for **non-commercial**, **scientific research**, and **educational purposes only**. Any use of this dataset for **commercial products**, **services**, or **other profit-oriented activities** is strictly prohibited.

### Content and Copyright

The audiovisual data in this dataset was collected from publicly available sources on the internet. All intellectual property rights, including copyright, for the original content belong to their respective owners. The compilation of this dataset is for **research** and **transformative use** only.

### Takedown Policy

We have made a best-effort attempt to curate this dataset for research purposes. If you are the copyright holder of any content included in **SpeakerVid-5M** and believe its inclusion infringes upon your rights or violates privacy, please contact us immediately at **[zhangyou24@mails.tsinghua.edu.cn]**.

Upon receiving a valid request, we will promptly investigate and remove the specified content from the dataset without delay.

### Disclaimer

The **SpeakerVid-5M** dataset is provided **"as-is"** without any express or implied warranties. The authors and their affiliated institutions are not liable for any damages or losses arising from the use of this dataset. Users are solely responsible for ensuring that their use of the data complies with all applicable laws and ethical guidelines.

---

For any inquiries, please contact us at **[zhangyou24@mails.tsinghua.edu.cn]**.
