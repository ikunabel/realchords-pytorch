# NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms

Yashan Wang<sup>1</sup><sup>∗</sup> , Shangda Wu<sup>1</sup><sup>∗</sup> , Jianhuai Hu<sup>1</sup><sup>∗</sup> , Xingjian Du<sup>2</sup> , Yueqi Peng<sup>3</sup> , Yongxin Huang<sup>4</sup> , Shuai Fan<sup>5</sup> , Xiaobing Li<sup>1</sup> , Feng Yu<sup>1</sup> , Maosong Sun<sup>1</sup>,6† <sup>1</sup>Central Conservatory of Music, China , <sup>2</sup>University of Rochester, USA , <sup>3</sup>Beijing Flowingtech Ltd., China , <sup>4</sup> Independent Researcher , <sup>5</sup>Beihang University, China , <sup>6</sup>Tsinghua University, China

{alexis\_wang, shangda, hujianhuai}@mail.ccom.edu.cn, sms@tsinghua.edu.cn

<https://electricalexis.github.io/notagen-demo>

# Abstract

We introduce NotaGen, a symbolic music generation model aiming to explore the potential of producing high-quality classical sheet music. Inspired by the success of Large Language Models (LLMs), NotaGen adopts pre-training, fine-tuning, and reinforcement learning paradigms (henceforth referred to as the LLM training paradigms). It is pre-trained on 1.6M pieces of music in ABC notation, and then fine-tuned on approximately 9K high-quality classical compositions conditioned on "period-composer-instrumentation" prompts. For reinforcement learning, we propose the CLaMP-DPO method, which further enhances generation quality and controllability without requiring human annotations or predefined rewards. Our experiments demonstrate the efficacy of CLaMP-DPO in symbolic music generation models with different architectures and encoding schemes. Furthermore, subjective A/B tests show that NotaGen outperforms baseline models against human compositions, greatly advancing musical aesthetics in symbolic music generation.

# 1 Introduction

The pursuit of musicality is a core objective in music generation research, as it fundamentally shapes how we perceive and experience musical compositions. Symbolic music abstracts music into discrete symbols such as notes and beats, with performance signals (i.e., MIDI) and sheet music (e.g., ABC notation, MusicXML) being the two predominant modalities. Both of them can efficiently model melody, harmony, instrumentation, etc., all of which are crucial for musicality.

Training tokenized representations with language model architectures, such as Transformers [\[Vaswani](#page-8-0) *et al.*, 2017], has emerged as a powerful paradigm for symbolic music generation [\[Huang](#page-7-0) *et al.*, 2018; [Casini and Sturm, 2022\]](#page-7-1). However, several challenges persist. First, the scarcity of high-

![](_page_0_Diagram_5.jpeg)

**Step 3: Reinforcement Learning (CLaMP-DPO)**

Figure 1: An overview of NotaGen's training paradigms.

quality music data [\[Hentschel](#page-7-2) *et al.*, 2023] hinders the ability of deep learning models to generate sophisticated compositions. Second, when optimizing a language model's loss function, the focus typically lies in minimizing the discrepancy between the predicted and the ground-truth next tokens, potentially neglecting holistic musical aspects like music structure and stylistic coherence.

Insights from the Natural Language Processing (NLP) domain provide a promising approach to overcoming the challenges inherent in symbolic music generation. The success of Large Language Models (LLMs) [\[Dubey](#page-7-3) *et al.*, 2024] has established the paradigm of pre-training, fine-tuning, and reinforcement learning as a widely acknowledged framework to improve the quality of text generation and align output with human preferences. These techniques have been successfully adapted for music generation. To overcome the scarcity of high-quality data, large-scale pre-training followed by finetuning on smaller, task-specific datasets has been employed

<sup>\*</sup>These authors contributed equally.

<sup>†</sup>Corresponding author.

effectively [\[Donahue](#page-7-4) *et al.*, 2019; Wu *et al.*[, 2024a\]](#page-8-1). Reinforcement Learning from Human Feedback (RLHF) [\[Sti](#page-8-2)ennon *et al.*[, 2020\]](#page-8-2), transcending next-token prediction approaches, has also shown promising results in music generation [\[Cideron](#page-7-5) *et al.*, 2024]. However, to the best of our knowledge, the complete pipeline of LLM training paradigms has not been fully implemented in symbolic music generation. Furthermore, the high cost of RLHF for human annotation highlights the necessity for more efficient and automated solutions.

In this work, we introduce NotaGen (Musical Notation Generation), a symbolic music generation model focused on classical sheet music. Compared to MIDI generation, sheet music generation not only aims to produce artistically refined music, but also emphasizes proper voice arrangement and notation to create well-formatted sheets for performance and analysis. Furthermore, the challenge of sheet music generation is exacerbated by the diverse instrumentation and rich musicality inherent in classical music. The success of LLMs has motivated us to apply the training paradigms to sheet music generation. NotaGen is pre-trained on a corpus of over 1.6M sheets in ABC notation, and fine-tuned on a collection of approximately 9K high-quality classical pieces from 152 composers with "period-composer-instrumentation" (e.g."Baroque-Bach, Johann Sebastian-Keyboard") prompts guiding conditional generation. During reinforcement learning, we introduce the CLaMP-DPO method to further optimize NotaGen's musicality and controllability using the Direct Preference Optimization (DPO) [\[Rafailov](#page-8-3) *et al.*, 2024] algorithm. In this approach, CLaMP 2 [Wu *et al.*[, 2024b\]](#page-8-4), a multimodal symbolic music information retrieval model, assigns generated samples as "chosen" or "rejected" based on references from the finetuning dataset. Our contributions are two-fold:

- We introduce NotaGen, a symbolic music generation model implementing LLM training paradigms, which significantly outperforms baseline models in subjective A/B tests against human compositions.
- We propose CLaMP-DPO, a reinforcement learning approach that integrates the DPO algorithm with CLaMP 2 feedback, enhancing musicality and controllability of symbolic music generation without relying on human annotation or predefined rewards. This potential is showcased across symbolic music generation models with varying architectures and encoding schemes.

## 2 Related Works

## 2.1 Sheet Music Generation

Sheet music generation has been widely studied, with a focus on encoding methods and composition modeling. Score Transformer [\[Suzuki, 2021\]](#page-8-5) introduces a tokenized representation for sheet music and applies it to piano music generation. Measure by Measure [\[Yan and Duan, 2024\]](#page-8-6) models sheet music as grids of part-wise bars and employs hierarchical architectures for generation. Compared to the intricate representations used by the models above, ABC notation, a comprehensive text-based sheet music representation, simplifies encoding and facilitates composition modeling, gaining

increasing adoption in recent research. The following models utilize the ABC notation: FolkRNN [Sturm *et al.*[, 2016\]](#page-8-7) and Tunesformer [Wu *et al.*[, 2023b\]](#page-8-8), specializing in folk melody generation; DeepChoir [Wu *et al.*[, 2023a\]](#page-8-9), which generates choral music with chord conditioning; and MuPT [Qu *[et al.](#page-8-10)*, [2024\]](#page-8-10), a large-scale pre-trained model for sheet music, which explores multitrack symbolic music generation.

#### 2.2 Pre-training in Symbolic Music Generation

The success of pre-training in NLP has inspired the application of this technique in symbolic music generation. LakhNES [\[Donahue](#page-7-4) *et al.*, 2019] enhances chiptune music generation by pre-training on the Lakh MIDI Dataset [\[Raf](#page-8-11)[fel, 2016\]](#page-8-11). MuseBERT [\[Wang and Xia, 2021\]](#page-8-12) adopts masked language modeling [\[Devlin](#page-7-6) *et al.*, 2019], while MelodyGLM [Wu *et al.*[, 2023d\]](#page-8-13) implements auto-regressive blank infilling [Du *et al.*[, 2021\]](#page-7-7) for generation. MelodyT5 [Wu *et al.*[, 2024a\]](#page-8-1) leverages multi-task learning [Raffel *et al.*[, 2020\]](#page-8-14). These studies highlight the effectiveness of pre-training in enhancing music generation performance.

## 2.3 Reinforcement Learning in Music Generation

Reinforcement learning has long been recognized as a promising approach for enhancing the musicality of music generation models. It has been successfully applied in RL Tuner [\[Jaques](#page-7-8) *et al.*, 2017] for melody generation, RL-Duet [Jiang *et al.*[, 2020\]](#page-7-9) for online duet accompaniment, RL-Chord [Ji *et al.*[, 2023\]](#page-7-10) for melody harmonization, and [\[Guo](#page-7-11) *et al.*[, 2022\]](#page-7-11) for multi-track music generation. However, these methods either base their rewards on music theory, which limits flexibility, or tailor them to specific music styles, hindering their generalization to a broader range of music generation tasks. To tackle this problem, MusicRL [\[Cideron](#page-7-5) *et al.*, 2024] adopts the RLHF method with extensive human feedback to align the generated compositions with human preference.

## 3 NotaGen

## 3.1 Data Representation

ABC notation sheets consist of two parts: the tune header, which contains metadata such as tempo, time signature, key, and instrumentation; the tune body, where the musical content for each voice is recorded. We adopt a modified version—interleaved ABC notation [Wu *et al.*[, 2024b;](#page-8-4) Qu *et al.*[, 2024\]](#page-8-10). In this format, different voices of the same bar are rearranged into a single line and differentiated using voice indicators "[V:]". This ensures alignment of duration and musical content across voices. Furthermore, we remove bars with full rests (containing only "z" or "x" notes), reducing the length to 80.7% on average, while increasing information density.

We employ stream-based training and inference methods to enable long musical piece generation. We annotate the current and countdown bar indices before each tune body line using the label "[r:]". During training, we randomly segment the tune body and concatenated it with the tune header for longer pieces; during inference, we enforce the generation to start from scratch using the bar annotations. If the piece is incomplete within the current context length, we concatenate

<span id="page-2-0"></span>![](_page_2_Diagram_0.jpeg)

Figure 2: Data representation and model architecture of NotaGen. (a) An example of data representation for an excerpt from *String Quartet in B-flat major, Hob.III:1* by Joseph Haydn using interleaved ABC notation. Bar annotations "[r:]" denote current/countdown bar indices, with gray bars representing omitted rests. Colored backgrounds delineate bar-stream patch boundaries. (b) The model architecture of NotaGen. After passing through the linear projection, bar-stream patches are processed by the patch-level decoder to generate features for a characterlevel decoder, which performs auto-regressive character prediction.

the generated tune header with the second half of the tune body and continue generating until the final bar.

## 3.2 Model Architecture

NotaGen utilizes bar-stream patching [Wang *et al.*[, 2024\]](#page-8-15) and the Tunesformer architecture [Wu *et al.*[, 2023b\]](#page-8-8). Building upon bar patching [Wu *et al.*[, 2023c\]](#page-8-16), bar-stream patching divides the tune header lines and bars into fixed-length patches (padded when necessary), striking a balance between musicality of generation and computational efficiency among sheet music tokenization methods.

NotaGen consists of two hierarchical GPT-2 decoders [\[Radford](#page-8-17) *et al.*, 2019]: a patch-level decoder and a characterlevel decoder. Each patch is flattened by concatenating onehot character vectors and then passed through a linear layer to obtain the patch embedding. The patch-level decoder captures the temporal relationships among patches, and its final hidden states are passed to the character-level decoder, which auto-regressively predicts the characters of the next patch. The data representation and model architecture are illustrated in Figure [2.](#page-2-0)

## 3.3 Training Paradigms

## Pre-training

Pre-training enables NotaGen to capture fundamental musical structures and patterns through next-token prediction on a large, diverse dataset spanning various genres and instrumentations.

The pre-training stage utilized a carefully curated internaluse dataset comprising 1.6M ABC notation sheets. We also preprocessed the text annotations, retaining music-related content such as tempo and expression hints, while removing irrelevant content like lyrics and background information.

All music sheets were transposed to 15 keys (including F♯, G♭, C♯, C♭) for data augmentation. During training, a randomly selected key was used for each piece in every epoch.

#### Fine-tuning

NotaGen was fine-tuned on high-quality classical music sheet data to further enhance musicality in generation. Spanning from the intricate contrapuntal orchestra suites of the Baroque period to the melodious and harmonically nuanced piano pieces of the Romantic era, classical music encompasses a diverse array of compositional styles and instrumentations, all characterized by exceptional musicality.

Thus, we curated a fine-tuning dataset comprising 8,948 classical music sheets, from DCML corpora [\[Neuwirth](#page-8-18) *et al.*[, 2018;](#page-8-18) [Hentschel](#page-7-12) *et al.*, 2021a; [Hentschel](#page-7-13) *et al.*, 2021b; [Hentschel](#page-7-2) *et al.*, 2023], OpenScore String Quartet Corpus [\[Gotham](#page-7-14) *et al.*, 2023], OpenScore Lieder Corpus [\[Gotham](#page-7-15) [and Jonas, 2022\]](#page-7-15), ATEPP [Zhang *et al.*[, 2022\]](#page-8-19), KernScores [\[Sapp, 2005\]](#page-8-20), and internal resources, as listed in Table [1.](#page-3-0) Sheets with more than 16 staves were excluded due to generation complexity. Each work was assigned with three labels: period, composer and instrumentation. The data distribution is provided in supplementary materials, and the details of each label are explained as follows:

## • Period:

- Baroque (1600s-1750s): e.g., Bach, Vivaldi.
- Classical (1750s-1810s): e.g., Mozart, Beethoven.
- Romantic (1810s-1950s): e.g., Chopin, Liszt.
- Composer: The official names of a total of 152 composers, as listed on IMSLP[<sup>1</sup>](#page-2-1) , were included.
- Instrumentation:
  - Keyboard: piano and organ works.
  - Chamber: instrumental music typically for a small group of performers, each playing a unique part.
  - Orchestral: instrumental music for orchestra.

<span id="page-2-1"></span><sup>1</sup> <https://imslp.org/>

<span id="page-3-0"></span>

| Data       | Sources        | Amount     |
|------------|----------------|------------|
| DCML       | Corpora        | 560        |
| OpenScore  | String Quartet | Corpus 342 |
| OpenScore  | Lieder Corpus  | 1,334      |
| ATEPP      |                | 55         |
| KernScores |                | 221        |
| Internal   | Sources        | 6,436      |
| Total      |                | 8,948      |

Table 1: Data sources and the respective amounts for fine-tuning.

- Art Song: vocal music typically for solo or duet voices with piano accompaniment.
- Choral: vocal music for a choir.
- Vocal-Orchestral: works involving both vocal and orchestral elements, including Cantata, Oratorio, and Opera.

In fine-tuning, a "period-composer-instrumentation" prompt was prepended to each piece for conditional generation. This approach challenges NotaGen to produce high-quality compositions, imitate the styles of composers across different periods, and conform to specified instrumentation requirements.

To facilitate NotaGen's learning of appropriate pitch ranges for each instrument while optimizing data utilization, data augmentation during fine-tuning was restricted to the six nearest key transpositions of the original. Keys farther from the original were selected with decreasing probability.

#### Reinforcement Learning

To refine both the musicality and the prompt controllability of the fine-tuned NotaGen, we present CLaMP-DPO. This method builds upon the principles of Reinforcement Learning from AI Feedback (RLAIF) [Lee *et al.*[, 2024\]](#page-7-16) and implements Direct Preference Optimization (DPO) [\[Rafailov](#page-8-3) *et al.*[, 2024\]](#page-8-3). In CLaMP-DPO, CLaMP 2 serves as the evaluator within the DPO framework, distinguishing between chosen and rejected musical outputs to optimize NotaGen.

CLaMP 2 is a multimodal symbolic music information retrieval model supporting both ABC notation and MIDI formats. Leveraging contrastive learning, CLaMP 2 extracts semantic features that encapsulate global musical properties. These features encompass comprehensive musical information, including style, instrumentation, and compositional complexity. Meanwhile, they are consistent with human subjective perceptions, as validated by [\[Retkowski](#page-8-21) *et al.*, 2024]. In the context of music generation, the objective is to produce pieces which closely resemble the ground truth. Accordingly, it is critical to ensure the alignment of the semantic features between the generated pieces and authentic references.

We introduce the CLaMP 2 Score to quantify the similarity among pieces. To elaborate, we denote P as the set of prompts for NotaGen. For each prompt p ∈ P, Y<sup>p</sup> represents the corresponding set of ground truth with an average semantic feature z¯p. Similarly, each prompt p has a generated set Xp, where each piece x<sup>p</sup> is associated with a semantic feature z<sup>x</sup><sup>p</sup> .

The CLaMP 2 score c for a generated piece x<sup>p</sup> is defined as the cosine similarity between z<sup>x</sup><sup>p</sup> and z¯p:

<span id="page-3-1"></span>c<sup>x</sup><sup>p</sup> = z<sup>x</sup><sup>p</sup> · z¯<sup>p</sup> ∥z<sup>x</sup><sup>p</sup> ∥∥z¯p∥ . (1)

Our goal is to maximize the average, c¯<sup>x</sup><sup>p</sup> over Xp, thereby ensuring the music generated for prompt p aligns semantically with the ground truth. It is achieved by employing the DPO algorithm to improve c¯<sup>x</sup><sup>p</sup> .

The DPO algorithm optimizes a language model based on preference data, which consists of paired chosen and rejected examples under the same prompts. It eliminates the need of explicit reward modeling. In the proposed CLaMP-DPO algorithm, the fine-tuned model first generates data across the prompt set P. For each generated set Xp, the pieces x<sup>p</sup> ∈ X<sup>p</sup> are sorted according to c<sup>x</sup><sup>p</sup> , with the top 10% selected as chosen set Xpw and the bottom 10% as rejected set Xpl. Additional criteria, such as syntax error checks or the exclusion of ground-truth plagiarism, can be applied to refine these sets. Finally, the chosen and rejected pairs (xpw, xpl) are randomly selected and combined into preference data for optimization.

Given a prompt p, an auto-regressive language model predicts the next token based on its policy πθ, where θ represents the model parameters. The probability of generating a chosen data xpw is πθ(xpw|p), and that of generating a rejected data xpl is πθ(xpl|p). To prevent excessive drift from the initial model that generates the preference data and ensure diversity in the generated content, the initial model policy, i.e., the reference model policy πref is introduced and kept frozen during optimization. The objective function to be minimized in DPO is given by:

<sup>L</sup>DPO(πθ; <sup>π</sup>ref) = <sup>−</sup>E(p,xpw,xpl)∼Dh log σ <sup>β</sup> log <sup>π</sup>θ(xpw|p) πref(xpw|p) <sup>−</sup> <sup>β</sup> log <sup>π</sup>θ(xpl|p) πref(xpl|p) i, (2)

where σ is the sigmoid function, D is the preference dataset, and β is the hyperparameter that controls the deviation between π<sup>θ</sup> and πref.

The optimization process increases the relative log probability of chosen data over rejected data. However, we observed a decrease in πθ(xpw|p), leading to degraded outputs. To mitigate this issue, we adopt the DPO-Positive (DPOP) objective function [Pal *et al.*[, 2024\]](#page-8-22), which incorporates a penalty term to stabilize πθ(xpw|p):

<span id="page-3-2"></span><sup>L</sup>DPOP(πθ; <sup>π</sup>ref) = <sup>−</sup>E(p,xpw,xpl)∼Dh log σ <sup>β</sup> log <sup>π</sup>θ(xpw|p) πref(xpw|p) <sup>−</sup> <sup>β</sup> log <sup>π</sup>θ(xpl|p) πref(xpl|p) <sup>−</sup> βλ · max <sup>0</sup>, log <sup>π</sup>ref(xpw|p) πθ(xpw|p) i, (3)

where the hyperparameter λ controls the impact of penalty.

#### Algorithm 1: Iterative CLaMP-DPO

Input: Fine-tuned policy π 0 θ , CLaMP 2 model C, prompt set P, fine-tuning dataset Y Parameter: Iterations K, DPO hyperparameter β, DPOP hyperparameter λ, optimization steps N, learning rate η Output: Optimized policy π K θ # Initialize ground-truth features <sup>1</sup> foreach prompt p ∈ P do <sup>2</sup> z¯<sup>p</sup> ← Avg(C(yp)), ∀y<sup>p</sup> ∈ Y<sup>p</sup> <sup>3</sup> end # Iterative Optimization <sup>4</sup> for k ← 1 to K do # Construct preference data <sup>5</sup> foreach prompt p ∈ P do <sup>6</sup> X<sup>k</sup>−<sup>1</sup> <sup>p</sup> ← π k−1 θ (p) # Generate on p <sup>7</sup> foreach piece x k−1 <sup>p</sup> ∈ X k−1 P do <sup>8</sup> z<sup>x</sup> k−1 <sup>p</sup> ← <sup>C</sup>(<sup>x</sup> k−1 p ) <sup>9</sup> c<sup>x</sup> k−1 <sup>p</sup> ← Eq. [\(1\)](#page-3-1)(z<sup>x</sup> k−1 p , z¯p) <sup>10</sup> end <sup>11</sup> X<sup>k</sup>−<sup>1</sup> pw , X<sup>k</sup>−<sup>1</sup> pl ← Select(X<sup>k</sup>−<sup>1</sup> p , Sort(c<sup>x</sup> k−1 )) <sup>12</sup> end # Optimize using DPO <sup>13</sup> πref ← π k−1 θ <sup>14</sup> for i ← 1 to N do <sup>15</sup> Sample prompt p ∼ P <sup>16</sup> Sample pairs (xpw, xpl) ∼ (X<sup>k</sup>−<sup>1</sup> pw , X<sup>k</sup>−<sup>1</sup> pl ) <sup>17</sup> θ ← θ − η∇θLDPOP(πθ, πref, xpw, xpl, p, β, λ) <sup>18</sup> end <sup>19</sup> π k <sup>θ</sup> ← π<sup>θ</sup> <sup>20</sup> end <sup>21</sup> return π K θ

The fine-tuned model is optimized by minimizing LDPOP in Eq.[\(3\)](#page-3-2) for a specified number of steps, completing the process of CLaMP-DPO algorithm. Notably, CLaMP-DPO supports iterative optimization. After the first round, the model generates a new set X′ p . Using CLaMP 2, we construct new chosen and rejected sets, X′ pl and <sup>X</sup>′ pw, allowing the model to undergo further optimization via Eq.[\(3\)](#page-3-2).

## 4 Experiments

### 4.1 Settings

The experiments are divided into two parts. The first part assesses CLaMP-DPO's ability to improve the controllability and musicality of symbolic music models. The second part compares the musicality of NotaGen with baseline models. Along with the pre-trained NotaGen, we selected two pre-trained symbolic music generation models as baselines: MuPT [Qu *et al.*[, 2024\]](#page-8-10) and Music Event Transformer (MET) [2](#page-4-0) [\[SkyTNT, 2024\]](#page-8-23). All models adopt language model architectures and are trained auto-regressively. A brief overview of their architectures and pre-training procedures follows:

- NotaGen features a 20-layer patch-level decoder and a 6-layer character-level decoder, with a context length of 1024 and a hidden size of 1280, totaling 516M parameters. It was pre-trained on 1.6M ABC notation sheets, augmented to 15 key transpositions. The AdamW optimizer [\[Loshchilov and Hutter, 2019\]](#page-8-24) was utilized with a learning rate of 1e-4 and a 1,000-step warm-up phase. The pre-training was performed on 8 NVIDIA H800 GPUs, with a batch size of 4 per GPU.
- MuPT utilizes Synchronized Multi-Track ABC notation (SMT-ABC) as data representation. SMT-ABC is equivalent to interleaved ABC notation, as both merge multitrack voices into a single sequence. Byte Pair Encoding (BPE) is used for tokenization. MuPT-v1-8192-550M, the chosen baseline model, consists of a 16-layer Transformer decoder with a hidden size of 1024 and a context length of 8192, totaling 505M parameters. MuPT was pre-trained on a corpus of 33.6B tokens.
- MET encodes MIDI events into token sequences and uses hierarchical Transformer decoders for generation, including a event-level decoder and a token-level decoder. Details on the encoding and model architecture are provided in the supplementary materials. MET consists of a 12-layer event-level decoder and a 3-layer token-level decoder, with a context length of 4096 and a hidden size of 1024, totaling 234M parameters. It was pre-trained on three MIDI datasets: Los Angeles MIDI Dataset [\[Lev, 2024\]](#page-7-17), Monster MIDI Dataset [<sup>3</sup>](#page-4-1) , and SymphonyNet Dataset [Liu *et al.*[, 2022\]](#page-7-18).

We applied fine-tuning and reinforcement learning to these models using their pre-trained weights.

Fine-tuning. The fine-tuning dataset for NotaGen and MuPT comprises 8,948 classical music pieces, referred to as the sheet ground truth set (sheet-GT). All data were formatted to match the pre-training data of different models, each preceded by a "period-composer-instrumentation" prompt. Due to the challenges in converting between MIDI and ABC notation, only the keyboard subset, consisting of 3,104 pieces was used for fine-tuning MET, referred to as the MIDI ground truth set (MIDI-GT). Each piece was preceded by a "periodcomposer" prompt.

Reinforcement learning. Considering that the accuracy of prompt semantic feature z¯<sup>p</sup> in CLaMP-DPO relies on a sufficient amount of ground truth data in Yp, we defined the prompt set P to only include prompts p that appear more than ten times in the fine-tuning dataset (Y<sup>p</sup> > 10). The detailed list of P can be referred in supplementary materials. For NotaGen and MuPT, P contained 112 prompts, covering 86.4% of the data; for MET, P contained 29 prompts, covering 90.5%. The number of iterations K was set to 3, with approximately 100 pieces generated per prompt as X<sup>p</sup> in each iteration. The chosen and rejected sets were constructed

<span id="page-4-0"></span><sup>2</sup> <https://huggingface.co/skytnt/midi-model-tv2o-medium>

<span id="page-4-1"></span>[https://huggingface.co/datasets/projectlosangeles/](https://huggingface.co/datasets/projectlosangeles/Monster-MIDI-Dataset) [Monster-MIDI-Dataset](https://huggingface.co/datasets/projectlosangeles/Monster-MIDI-Dataset)

based on CLaMP 2 Scores. Sheets where staves for the same instrument were not grouped together were excluded from the chosen set. The hyperparameters β = 0.1 and λ = 10 were used, with N = 10, 000 optimization steps. The learning rate was fixed at 1e-6 for NotaGen and MET, and 1e-7 for MuPT, yielding stable CLaMP-DPO performance.

Given the challenge of establishing objective metrics that fully capture musicality, we conducted subjective A/B tests in both experiments to evaluate different models and settings. For each question, two pieces were generated using identical prompts; videos were rendered from sheet music using Sibelius and MIDI files using MIDIVisualizer[<sup>4</sup>](#page-5-0) . Participants were instructed to evaluate musicality from multiple perspectives and select the piece they found more musically appealing, or indicate no preference if they perceived no differences. The evaluation criteria included melodic appeal, harmonic fluency, orchestral balance, counterpoint correctness, and structural coherence, and, for sheet music, notation formatting quality. A total of 92 participants from music colleges took part in the assessment. At least 35 valid responses were recorded for each test group, ensuring statistical reliability.

## 4.2 Ablation Studies on CLaMP-DPO

This experiment evaluates the impact of the proposed CLaMP-DPO algorithm in enhancing the controllability and musicality of generated music for NotaGen, MuPT, and MET. In the objective assessment, we selected several metrics for both the fine-tuned models (denoted as K = 0) and the models after K iterations of CLaMP-DPO optimization. We also assessed a subset of these metrics on the fine-tuning datasets (sheet-GT and MIDI-GT) for reference. The metrics are as follows:

- Average CLaMP 2 Score (ACS): The average CLaMP 2 Score across generated outputs. For sheet-GT and MIDI-GT, the score is computed over the corpus.
- Label Accuracy (LA): The alignment with specified period (per.) and instrumentation (inst.) prompts. We extracted features from the fine-tuning dataset via a multimodal symbolic music encoder—M3[Wu *et al.*[, 2024b\]](#page-8-4), then trained two linear classifiers to predict the period and instrumentation labels. LA is defined as the classification accuracy, where for the fine-tuning dataset, it measures the accuracy on the test set, and for generated outputs, it reflects the match between predicted labels and prompt labels.
- Bar Alignment Error (BAE): The proportion of bars where duration is misaligned, occurring in either the generated output or the fine-tuning corpus. This metric applies only to sheet data.
- Perplexity (PPL): A language model metric, where lower PPL indicates better prediction capability.

We conducted subjective A/B tests on each model before and after three optimization iterations with CLaMP-DPO to

<span id="page-5-1"></span>

| Models & Data | K | ACS   | LA Per. | (%) Inst. | BAE (%) | PPL    |
|---------------|---|-------|---------|-----------|---------|--------|
| sheet-GT      |   | 0.792 | 96.1    | 95.5      | 0.377   |        |
|               | 0 | 0.570 | 84.7    | 78.5      | 0.269   | 1.2151 |
| NotaGen       | 1 | 0.674 | 92.1    | 87.8      | 0.175   | 1.2341 |
|               | 2 | 0.708 | 93.3    | 92.9      | 0.158   | 1.2614 |
|               | 3 | 0.730 | 93.0    | 94.6      | 0.176   | 1.2880 |
|               | 0 | 0.515 | 76.3    | 78.6      | 0.824   | 1.4159 |
| MuPT          | 1 | 0.596 | 78.8    | 86.2      | 1.520   | 1.4476 |
|               | 2 | 0.631 | 80.3    | 89.2      | 2.601   | 1.5214 |
|               | 3 | 0.674 | 82.1    | 87.6      | 4.676   | 1.6121 |
| MIDI-GT       |   | 0.812 | 92.9    |           |         |        |
|               | 0 | 0.565 | 30.0    |           |         | 1.2251 |
| MET           | 1 | 0.609 | 34.6    |           |         | 1.2261 |
|               | 2 | 0.637 | 36.7    |           |         | 1.2255 |
|               | 3 | 0.655 | 38.2    |           |         | 1.2290 |

Table 2: Objective metrics on fine-tuned models and the models after each iteration of CLaMP-DPO optimization. Some of metrics were also assessed on the fine-tuning dataset for reference.

<span id="page-5-2"></span>![](_page_5_Figure_4.jpeg)

Figure 3: Subjective A/B tests on musicality of generated outputs before and after CLaMP-DPO optimization. All models exhibited improvement in human-perveiced musicality after applying the CLaMP-DPO algorithm.

appraise its efficacy in enhancing the musical quality of generated outputs. The results of the objective and subjective tests are presented in Table [2](#page-5-1) and Figure [3,](#page-5-2) respectively.

The ACS, as the primary optimization goal, exhibited a monotonic increase across iterations of its DPO-based process. Though significant improvements were observed in early iterations, subsequent gains exhibited diminishing returns.

LA for period and instrumentation classification exceeded 90% on the test set of fine-tuning data, validating the reliability of the label assignments and the performance of the classifiers. Following the CLaMP-DPO method, all models demonstrated a noticeable improvement in LA, indicating enhanced prompt controllability and better alignment with the intended musical styles. NotaGen exhibited the highest con-

<span id="page-5-0"></span><sup>4</sup> <https://github.com/kosua20/MIDIVisualizer>

trollability among the models, further confirming its superior adaptability to specified prompts.

Regarding BAE, NotaGen maintained a relatively low error rate throughout optimization, indicating its character-level prediction is more robust at managing duration consistency. In contrast, MuPT's increased error rate is likely due to the use of BPE tokenization, which may merge duration with other musical elements, such as pitch, into single tokens. It may lead to inaccuracies in duration prediction after CLaMP-DPO adjusts token probabilities.

Subjective A/B tests showed that all models exhibited improvement in musicality after applying the CLaMP-DPO algorithm, with post-optimization outputs receiving more votes than their pre-optimization counterparts. However, it is noteworthy that PPL increased after optimization. It suggests that PPL may not be a suitable indicator for model performance in symbolic music generation, highlighting the limitations of traditional language model metrics in assessing musical quality.

In summary, the CLaMP-DPO algorithm efficiently enhanced both the controllability and the musicality across three models, irrespective of their data modalities, encoding schemes, or model architectures. This underscores CLaMP-DPO's broad applicability and potential for auto-regressively trained symbolic music generation models.

### 4.3 Comparative Evaluations

This experiment compares the musicality of three models after the LLM training paradigm. For baseline comparison, we constructed the reference set using human-authored musical pieces from the fine-tuning dataset, which represent professional compositional standards. The subjective A/B tests were organized into three groups, each containing the generated results of a model and the ground truth. For comparison involving MET, all data were converted to MIDI to eliminate format-based bias. The results are shown in Figure [4.](#page-6-0)

Human compositions consistently outperformed all modelgenerated outputs in voting due to their exceptional musicality. Nevertheless, NotaGen achieved the highest voting rate against the ground truth among the three models, suggesting its superior perceived musicality relative to other systems in human evaluations.

Overall, NotaGen outperformed the baseline models. The superior performance of NotaGen compared to MuPT is attributed to well-designed data representation and tokenization. Despite its architectural similarities to MET, NotaGen achieved better musicality, benefiting from the efficiency and structural integrity of sheet music representation compared to MIDI.

## 5 Limitations and Challenges

While NotaGen shows promising advancements in symbolic music generation, limitations and challenges still warrant discussion.

We once introduced a post-training stage between pretraining and fine-tuning, refining the model with classicalstyle subset of the pre-training dataset. While it accelerated the fine-tuning convergence and improved ACS for NotaGen, the impact was less pronounced on MuPT and MET.

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

Figure 4: Subjective A/B test between model outputs and ground truth. NotaGen achieved the highest voting rate against the ground truth among the three models.

Furthermore, the prerequisite for evaluating generated results using CLaMP 2 Score is that the model has been well trained and is capable of generating reasonable compositions. For corrupted or syntactically flawed pieces, the CLaMP 2 Score may not reliably indicate the true musical similarity.

Finally, we found that modeling orchestral music presents greater challenges compared to smaller ensembles (e.g. solo piano or string quartets). While rest-bar omission during data pre-processing addresses the degeneration due to excessive blank bars in ensemble writing, NotaGen's performance in orchestral music generation still lags behind. More effective methods are expected for generating large ensemble compositions.

## 6 Conclusions

In this work, we present NotaGen, a symbolic music generation model designed to advance the musicality of generated outputs through a comprehensive LLM-inspired training paradigm. By integrating pre-training, fine-tuning, and reinforcement learning with the proposed CLaMP-DPO algorithm, NotaGen demonstrates superior performance in generating compositions that align with both the music style specified by prompts and human-perceived musicality. Our experiments validate two key findings: (1) CLaMP-DPO efficaciously enhances controllability and musicality across diverse symbolic music models, regardless of their modality, architectures, or encoding schemes, without requiring human annotations or predefined rewards; (2) NotaGen outperforms baseline models in subjective evaluations, achieving the highest voting rate against human-composed ground truth.

NotaGen establishes the viability of adapting LLM training paradigms to symbolic music generation, while addressing domain-specific challenges, including data scarcity and demand for high-quality music outputs. Future work could extend this framework with CLaMP-DPO to broader musical genres such as jazz, pop, and ethnic music; as well as exploring its compatibility with emerging music generation models.

- <span id="page-7-11"></span>Acknowledgments We would like to express our sincere gratitude to SkyTNT, the author of MET, for his valuable support on this project. We also acknowledge Yuling Yang, Xinran Zhang, Jiafeng Liu, Yuqing Cheng, and Yuhao Ding from Central Conservatory of Music for their support, especially on subjective tests and paper writing. This work was supported by the following funding sources: Special Program of National Natural Science Foundation of China (Grant No. T2341003), Advanced Discipline Construction Project of Beijing Universities, Major Program of National Social Science Fund of China (Grant No. 21ZD19), and the National Culture and Tourism Technological Innovation Engineering Project (Research and Application of 3D Music). References [Casini and Sturm, 2022] Luca Casini and Bob Sturm. Tradformer: A transformer model of traditional music transcriptions. In *International Joint Conference on Artificial Intelligence IJCAI 2022, Vienna, Austria, 23-29 July 2022*, pages 4915–4920, 2022. [Cideron *et al.*, 2024] Geoffrey Cideron, Sertan Girgin, Mauro Verzetti, Damien Vincent, Matej Kastelic, Zalán Borsos, Brian McWilliams, Victor Ungureanu, Olivier Bachem, Olivier Pietquin, et al. Musicrl: Aligning music generation to human preferences. *arXiv preprint arXiv:2402.04229*, 2024. [Devlin *et al.*, 2019] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of naacL-HLT*, volume 1, page 2. Minneapolis, Minnesota, 2019. [Donahue *et al.*, 2019] Chris Donahue, Huanru Henry Mao, Yiting Ethan Li, Garrison W Cottrell, and Julian McAuley. Lakhnes: Improving multi-instrumental music generation with cross-domain pre-training. *arXiv preprint arXiv:1907.04868*, 2019. [Du *et al.*, 2021] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. *arXiv preprint arXiv:2103.10360*, 2021. [Dubey *et al.*, 2024] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024. [Gotham and Jonas, 2022] Mark Robert Haigh Gotham and Peter Jonas. The openscore lieder corpus. In *Music Encoding Conference Proceedings 2021, 19–22 July, 2021 University of Alicante (Spain): Onsite & Online*, pages 131–136. Universidad de Alicante/Universitat d'Alacant, 2022. [Gotham *et al.*, 2023] Mark Gotham, Maureen Redbond, Bruno Bower, and Peter Jonas. The "openscore string quartet" corpus. In *Proceedings of the 10th International Conference on Digital Libraries for Musicology*, pages 49–57, 2023. [Guo *et al.*, 2022] Xuefei Guo, Hongguang Xu, and Ke Xu. Fine-tuning music generation with reinforcement learning based on transformer. In *2022 IEEE 16th International Conference on Anti-counterfeiting, Security, and Identification (ASID)*, pages 1–5. IEEE, 2022. [Hentschel *et al.*, 2021a] Johannes Hentschel, Fabian Claude Moss, Markus Neuwirth, and Martin Rohrmeier. A semi-automated workflow paradigm for the distributed creation and curation of expert annotations. In *Proceedings of the 22nd International Society for Music Information Retrieval Conference*, 2021. [Hentschel *et al.*, 2021b] Johannes Hentschel, Markus Neuwirth, and Martin Rohrmeier. The annotated mozart sonatas: Score, harmony, and cadence. *Transactions of the International Society for Music Information Retrieval*, 4(1):67–80, 2021. [Hentschel *et al.*, 2023] Johannes Hentschel, Yannis Rammos, Fabian C Moss, Markus Neuwirth, and Martin Rohrmeier. An annotated corpus of tonal piano music from the long 19th century. *Empirical Musicology Review*, 18(1):84–95, 2023. [Huang *et al.*, 2018] Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M Dai, Matthew D Hoffman, Monica Dinculescu, and Douglas Eck. Music transformer. *arXiv preprint arXiv:1809.04281*, 2018. [Jaques *et al.*, 2017] Natasha Jaques, Shixiang Gu, Richard E Turner, and Douglas Eck. Tuning recurrent neural networks with reinforcement learning. 2017. [Ji *et al.*, 2023] Shulei Ji, Xinyu Yang, Jing Luo, and Juan
  - Li. Rl-chord: Clstm-based melody harmonization using deep reinforcement learning. *IEEE Transactions on Neural Networks and Learning Systems*, 2023. [Jiang *et al.*, 2020] Nan Jiang, Sheng Jin, Zhiyao Duan, and Changshui Zhang. Rl-duet: Online music accompaniment generation using deep reinforcement learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 710–718, 2020. [Lee *et al.*, 2024] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. RLAIF vs. RLHF: scaling reinforcement learning from human feedback with AI feedback. In *Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024*. OpenReview.net, 2024. [Lev, 2024] Aleksandr Lev. Los angeles midi dataset: Sota kilo-scale midi dataset for mir and music ai purposes. In *GitHub*, 2024. [Liu *et al.*, 2022] Jiafeng Liu, Yuanliang Dong, Zehua Cheng, Xinran Zhang, Xiaobing Li, Feng Yu, and Maosong Sun. Symphony generation with permutation invariant language model. *arXiv preprint arXiv:2205.05448*, 2022.

<span id="page-7-18"></span><span id="page-7-17"></span><span id="page-7-16"></span><span id="page-7-15"></span><span id="page-7-14"></span><span id="page-7-13"></span><span id="page-7-12"></span><span id="page-7-10"></span><span id="page-7-9"></span><span id="page-7-8"></span><span id="page-7-7"></span><span id="page-7-6"></span><span id="page-7-5"></span><span id="page-7-4"></span><span id="page-7-3"></span><span id="page-7-2"></span><span id="page-7-1"></span><span id="page-7-0"></span>

<span id="page-8-24"></span><span id="page-8-23"></span><span id="page-8-22"></span><span id="page-8-21"></span><span id="page-8-20"></span><span id="page-8-19"></span><span id="page-8-18"></span><span id="page-8-17"></span><span id="page-8-16"></span><span id="page-8-15"></span><span id="page-8-14"></span><span id="page-8-13"></span><span id="page-8-12"></span><span id="page-8-11"></span><span id="page-8-10"></span><span id="page-8-9"></span><span id="page-8-8"></span><span id="page-8-7"></span><span id="page-8-6"></span><span id="page-8-5"></span><span id="page-8-4"></span><span id="page-8-3"></span><span id="page-8-2"></span><span id="page-8-1"></span><span id="page-8-0"></span>[Loshchilov and Hutter, 2019] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2019. [Neuwirth *et al.*, 2018] Markus Neuwirth, Daniel Harasim, Fabian C Moss, and Martin Rohrmeier. The annotated beethoven corpus (abc): A dataset of harmonic analyses of all beethoven string quartets. *Frontiers in Digital Humanities*, 5:379513, 2018. [Pal *et al.*, 2024] Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive. *arXiv preprint arXiv:2402.13228*, 2024. [Qu *et al.*, 2024] Xingwei Qu, Yuelin Bai, Yinghao Ma, Ziya Zhou, Ka Man Lo, Jiaheng Liu, Ruibin Yuan, Lejun Min, Xueling Liu, Tianyu Zhang, et al. Mupt: A generative symbolic music pretrained transformer. *arXiv preprint arXiv:2404.06393*, 2024. [Radford *et al.*, 2019] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019. [Rafailov *et al.*, 2024] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36, 2024. [Raffel *et al.*, 2020] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21(140):1– 67, 2020. [Raffel, 2016] Colin Raffel. *Learning-based methods for comparing sequences, with applications to audio-to-midi alignment and matching*. Columbia University, 2016. [Retkowski *et al.*, 2024] Jan Retkowski, Jakub St˛epniak, and Mateusz Modrzejewski. Frechet music distance: A metric for generative symbolic music evaluation. *arXiv preprint arXiv:2412.07948*, 2024. [Sapp, 2005] Craig Stuart Sapp. Online database of scores in the humdrum file format. In *ISMIR*, pages 664–665, 2005. [SkyTNT, 2024] SkyTNT. Midi model: Midi event transformer for symbolic music generation. [https://github.com/](https://github.com/SkyTNT/midi-model) [SkyTNT/midi-model,](https://github.com/SkyTNT/midi-model) 2024. [Stiennon *et al.*, 2020] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33:3008–3021, 2020. [Sturm *et al.*, 2016] Bob L Sturm, Joao Felipe Santos, Oded Ben-Tal, and Iryna Korshunova. Music transcription modelling and composition using deep learning. *arXiv preprint arXiv:1604.08723*, 2016. [Suzuki, 2021] Masahiro Suzuki. Score transformer: Generating musical score from note-level representation. In *Proceedings of the 3rd ACM International Conference on Multimedia in Asia*, pages 1–7, 2021. [Vaswani *et al.*, 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Lukasz Kaiser Jones, Aidan Nating, and Illia Gomez, & Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, pages 5998–6008. Curran Associates, Inc., 2017. [Wang and Xia, 2021] Ziyu Wang and Gus Xia. Musebert: Pre-training music representation for music understanding and controllable generation. In *ISMIR*, pages 722–729, 2021. [Wang *et al.*, 2024] Yashan Wang, Shangda Wu, Xingjian Du, and Maosong Sun. Exploring tokenization methods for multitrack sheet music generation. *arXiv preprint arXiv:2410.17584*, 2024. [Wu *et al.*, 2023a] Shangda Wu, Xiaobing Li, and Maosong Sun. Chord-conditioned melody harmonization with controllable harmonicity. In *ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 1–5. IEEE, 2023. [Wu *et al.*, 2023b] Shangda Wu, Xiaobing Li, Feng Yu, and Maosong Sun. Tunesformer: Forming irish tunes with control codes by bar patching. *arXiv preprint arXiv:2301.02884*, 2023. [Wu *et al.*, 2023c] Shangda Wu, Dingyao Yu, Xu Tan, and Maosong Sun. Clamp: Contrastive language-music pretraining for cross-modal symbolic music information retrieval. *arXiv preprint arXiv:2304.11029*, 2023. [Wu *et al.*, 2023d] Xinda Wu, Zhijie Huang, Kejun Zhang, Jiaxing Yu, Xu Tan, Tieyao Zhang, Zihao Wang, and Lingyun Sun. Melodyglm: multi-task pre-training for symbolic melody generation. *arXiv preprint arXiv:2309.10738*, 2023. [Wu *et al.*, 2024a] Shangda Wu, Yashan Wang, Xiaobing Li, Feng Yu, and Maosong Sun. Melodyt5: A unified scoreto-score transformer for symbolic music processing. *arXiv preprint arXiv:2407.02277*, 2024. [Wu *et al.*, 2024b] Shangda Wu, Yashan Wang, Ruibin Yuan, Zhancheng Guo, Xu Tan, Ge Zhang, Monan Zhou, Jing Chen, Xuefeng Mu, Yuejie Gao, et al. Clamp 2: Multimodal music information retrieval across 101 languages using large language models. *arXiv preprint arXiv:2410.13267*, 2024. [Yan and Duan, 2024] Yujia Yan and Zhiyao Duan. Measure by measure: Measure-based automatic music composition with modern staff notation. *Transactions of the International Society for Music Information Retrieval*, Nov 2024. [Zhang *et al.*, 2022] Huan Zhang, Jingjing Tang, Syed RM Rafee, Simon Dixon, George Fazekas, and Geraint A Wiggins. Atepp: A dataset of automatically transcribed expressive piano performance. In *Ismir 2022 Hybrid Conference*, 2022.

## A MIDI Event Transformer

MIDI Event Transformer (MET) is a pre-trained music generation model. It encodes MIDI events into token sequences and utilizes hierarchical transformer decoders for generation.

### A.1 MIDI Encoding

In the MIDI encoding process, each MIDI event is transformed into a token sequence s<sup>i</sup> . This sequence begins with the token representing the event type e<sup>i</sup> , followed by tokens corresponding to the event parameters p j i , as shown below:

s<sup>i</sup> = {e<sup>i</sup> , p<sup>1</sup> i , p<sup>2</sup> i , ..., p<sup>n</sup> <sup>i</sup> } (4)

The types of events and their corresponding parameters are listed in Table [3.](#page-9-0) Among these, BOS (beginning-ofsequence) and EOS (end-of-sequence) events are used to mark the start and end of a musical piece, respectively. Parameters such as channel (16 values), pitch (128 values), velocity (128 values), controller (128 values), program (128 values), controller value (128 values) all comply with the MIDI standard. The details of additional parameters are as follows:

- Time 1 and time 2: Represent the timing of an event. The absolute time position is calculated as the cumulative sum of ticks preceding the event, and the beat position is obtained by dividing this sum by the ticks per beat. Time 1 captures the beat difference between the current and previous events, with a vocabulary size of
  - 128. To enable finer-grained positioning, a beat is subdivided into 16 parts, with time 2 indicating which subdivision the event falls into.
- Track: Describes the track in which the event occurs, with a maximum of 128 possible tracks.
- Duration: The note duration is calculated using a resolution of 64th notes, with an upper limit of 2048, which allows for the representation of notes up to 128 beats long.
- BPM: Beats per minute, with a range from 1 to 384.
- Numerator and denominator: The numerator can range from 1 to 16, while the denominator can be one of 2, 4, 8 or 16.
- Key signature accidentals: Indicates the number of sharps or flats in the key signature. The value ranges from -7 (7 flats, C♭) to 7 (7 sharps, C♯).
- Mode: Specifies whether the key is major or minor, with values between 0 and 1.

Additionally, as a special token type for p j i , <PAD> is used to pad all sequences to a uniform length of 8. This results in a total vocabulary size of 3406 unique tokens.

During the fine-tuning and reinforcement learning stages of our experiments, we extend the original vocabulary by including 3 period IDs and 36 composer IDs. These are prepended as "period" and "composer" events—each represented by a sequence of 8 identical IDs—prior to the encoded MIDI sequences, serving as prompts.

<span id="page-9-0"></span>

| Event         | Parameters                                  |
|---------------|---------------------------------------------|
| Note          | time 1, time 2, track, channel, pitch,      |
|               | velocity, duration                          |
| Program       | Change time 1, time 2, track, channel,      |
| Control       | Change time 1, time 2, track, channel,      |
|               | controller, controller value                |
| Set Tempo     | time 1, time 2, track, bpm                  |
| Time          | Signature time 1, time 2, track, numerator, |
| Key Signature | time 1, time 2, track, key signature        |
|               | accidentals, mode                           |
| BOS           | <BOS>                                       |
| EOS           | <EOS>                                       |

Table 3: Event and parameter types in MIDI Event Transformer's encoding.

![](_page_9_Diagram_8.jpeg)

Figure 5: Illustration of the MIDI Event Transformer architecture, showcasing its two hierarchical decoders: the event-level decoder, which models temporal dependencies across high-level events, and the token-level decoder, which generates the detailed token sequence in an auto-regressive manner.

### A.2 Model Architecture

The MIDI Event Transformer is comprised of two hierarchical decoders: an event-level decoder and a token-level decoder, both of which are based on the Llama architecture.

Initially, each event token sequence is processed through a token embedding layer, where individual token embeddings are aggregated to produce a dense vector representation. The event-level decoder focuses on modeling temporal dependencies across high-level events, thereby capturing their sequential relationships. The output hidden states from the eventlevel decoder are subsequently passed into the token-level decoder, which generates the detailed token sequence in an auto-regressive manner.

# B Distribution of the Fine-tuning Dataset

We utilized CLaMP 2 to extract the semantic features of all pieces, and the distribution of each label group is shown in Figure [6](#page-10-0) (for composers, only eight composers with most pieces in the dataset are shown).

<span id="page-10-0"></span>Classical Romantic Baroque

(a) Distribution on periods.

![](_page_10_Figure_7.jpeg)

(b) Distribution on most frequent composers.

![](_page_10_Figure_9.jpeg)

(c) Distribution on instrumentations.

Figure 6: *t*-SNE visualizations for each label group on the finetuning dataset.

# C Prompt Set in Experiments

The full list of prompt set P in the reinforcement learning stage of our experiments is listed in Table [4.](#page-11-0)

| Period    | Composer       |                  |                | Instrumentation  |
|-----------|----------------|------------------|----------------|------------------|
| Baroque   | Bach,          | Johann Sebastian |                | Chamber          |
| Baroque   | Bach,          | Johann Sebastian |                | Choral           |
| Baroque   | Bach,          | Johann Sebastian |                | Keyboard         |
| Baroque   | Bach,          | Johann Sebastian |                | Orchestral       |
| Baroque   | Bach,          | Johann Sebastian |                | Vocal-Orchestral |
| Baroque   | Corelli,       | Arcangelo        |                | Chamber          |
| Baroque   | Corelli,       | Arcangelo        |                | Orchestral       |
| Baroque   | Handel,        | George           | Frideric       | Chamber          |
| Baroque   | Handel,        | George           | Frideric       | Keyboard         |
| Baroque   | Handel,        | George           | Frideric       | Orchestral       |
| Baroque   | Handel,        | George           | Frideric       | Vocal-Orchestral |
| Baroque   | Scarlatti,     | Domenico         |                | Keyboard         |
| Baroque   | Vivaldi,       | Antonio          |                | Chamber          |
| Baroque   | Vivaldi,       | Antonio          |                | Orchestral       |
| Baroque   | Vivaldi,       | Antonio          |                | Vocal-Orchestral |
| Classical | Beethoven,     | Ludwig           | van            | Art Song         |
| Classical | Beethoven,     | Ludwig           | van            | Chamber          |
| Classical | Beethoven,     | Ludwig           | van            | Keyboard         |
| Classical | Beethoven,     | Ludwig           | van            | Orchestral       |
| Classical | Haydn,         | Joseph           |                | Chamber          |
| Classical | Haydn,         | Joseph           |                | Keyboard         |
| Classical | Haydn,         | Joseph           |                | Orchestral       |
| Classical | Haydn,         | Joseph           |                | Vocal-Orchestral |
| Classical | Mozart,        | Wolfgang         | Amadeus        | Chamber          |
| Classical | Mozart,        | Wolfgang         | Amadeus        | Choral           |
| Classical | Mozart,        | Wolfgang         | Amadeus        | Keyboard         |
| Classical | Mozart,        | Wolfgang         | Amadeus        | Orchestral       |
| Classical | Mozart,        | Wolfgang         | Amadeus        | Vocal-Orchestral |
| Classical | Paradis,       | Maria            | Theresia von   | Art Song         |
| Classical | Reichardt,     | Louise           |                | Art Song         |
| Classical | Saint-Georges, |                  | Joseph Bologne | Chamber          |
| Classical | Schroter,      | Corona           |                | Art Song         |
| Romantic  | Bartok,        | Bela             |                | Keyboard         |
| Romantic  | Berlioz,       | Hector           |                | Choral           |
| Romantic  | Bizet,         | Georges          |                | Art Song         |
| Romantic  | Boulanger,     | Lili             |                | Art Song         |
| Romantic  | Boulton,       | Harold           |                | Art Song         |
| Romantic  | Brahms,        | Johannes         |                | Art Song         |
| Romantic  | Brahms,        | Johannes         |                | Chamber          |
| Romantic  | Brahms,        | Johannes         |                | Choral           |
| Romantic  | Brahms,        | Johannes         |                | Keyboard         |
| Romantic  | Brahms,        | Johannes         |                | Orchestral       |
| Romantic  | Burgmuller,    | Friedrich        |                | Keyboard         |
| Romantic  | Butterworth,   | George           |                | Art Song         |
| Romantic  | Chaminade,     | Cecile           |                | Art Song         |
| Romantic  | Chausson,      | Ernest           |                | Art Song         |
| Romantic  | Chopin,        | Frederic         |                | Art Song         |
| Romantic  | Chopin,        | Frederic         |                | Keyboard         |
| Romantic  | Cornelius,     | Peter            |                | Art Song         |
| Romantic  | Debussy,       | Claude           |                | Art Song         |
| Romantic  | Debussy,       | Claude           |                | Keyboard         |

Continued on next page

Table 2 – Continued

| Period   | Composer      |              | Instrumentation |
|----------|---------------|--------------|-----------------|
| Romantic | Dvorak,       | Antonin      | Chamber         |
| Romantic | Dvorak,       | Antonin      | Choral          |
| Romantic | Dvorak,       | Antonin      | Keyboard        |
| Romantic | Dvorak,       | Antonin      | Orchestral      |
| Romantic | Faisst,       | Clara        | Art Song        |
| Romantic | Faure,        | Gabriel      | Art Song        |
| Romantic | Faure,        | Gabriel      | Chamber         |
| Romantic | Faure,        | Gabriel      | Keyboard        |
| Romantic | Franz,        | Robert       | Art Song        |
| Romantic | Gonzaga,      | Chiquinha    | Art Song        |
| Romantic | Grandval,     | Clemence de  | Art Song        |
| Romantic | Grieg,        | Edvard       | Keyboard        |
| Romantic | Grieg,        | Edvard       | Orchestral      |
| Romantic | Hensel,       | Fanny        | Art Song        |
| Romantic | Holmes,       | Augusta Mary | Anne Art Song   |
| Romantic | Jaell,        | Marie        | Art Song        |
| Romantic | Kinkel,       | Johanna      | Art Song        |
| Romantic | Kralik,       | Mathilde     | Art Song        |
| Romantic | Lang,         | Josephine    | Art Song        |
| Romantic | Lehmann,      | Liza         | Art Song        |
| Romantic | Liszt,        | Franz        | Keyboard        |
| Romantic | Mayer,        | Emilie       | Chamber         |
| Romantic | Medtner,      | Nikolay      | Keyboard        |
| Romantic | Mendelssohn,  | Felix        | Art Song        |
| Romantic | Mendelssohn,  | Felix        | Chamber         |
| Romantic | Mendelssohn,  | Felix        | Choral          |
| Romantic | Mendelssohn,  | Felix        | Keyboard        |
| Romantic | Mendelssohn,  | Felix        | Orchestral      |
| Romantic | Munktell,     | Helena       | Art Song        |
| Romantic | Parratt,      | Walter       | Choral          |
| Romantic | Prokofiev,    | Sergey       | Keyboard        |
| Romantic | Rachmaninoff, | Sergei       | Choral          |
| Romantic | Rachmaninoff, | Sergei       | Keyboard        |
| Romantic | Ravel,        | Maurice      | Art Song        |
| Romantic | Ravel,        | Maurice      | Chamber         |
| Romantic | Ravel,        | Maurice      | Keyboard        |
| Romantic | Saint-Saens,  | Camille      | Chamber         |
| Romantic | Saint-Saens,  | Camille      | Keyboard        |
| Romantic | Saint-Saens,  | Camille      | Orchestral      |
| Romantic | Satie,        | Erik         | Art Song        |
| Romantic | Satie,        | Erik         | Keyboard        |
| Romantic | Schubert,     | Franz        | Art Song        |
| Romantic | Schubert,     | Franz        | Chamber         |
| Romantic | Schubert,     | Franz        | Choral          |
| Romantic | Schubert,     | Franz        | Keyboard        |
| Romantic | Schumann,     | Clara        | Art Song        |
| Romantic | Schumann,     | Robert       | Art Song        |
| Romantic | Schumann,     | Robert       | Chamber         |
| Romantic | Schumann,     | Robert       | Choral          |
| Romantic | Schumann,     | Robert       | Keyboard        |
| Romantic | Scriabin,     | Aleksandr    | Keyboard        |
| Romantic | Shostakovich, | Dmitry       | Chamber         |
| Romantic | Shostakovich, | Dmitry       | Keyboard        |
| Romantic | Sibelius,     | Jean         | Keyboard        |
| Romantic | Smetana,      | Bedrich      | Keyboard        |

<span id="page-11-0"></span>

|          |              |         | Table 2         | – Continued |
|----------|--------------|---------|-----------------|-------------|
| Period   | Composer     |         | Instrumentation |             |
| Romantic | Tchaikovsky, | Pyotr   | Keyboard        |             |
| Romantic | Tchaikovsky, | Pyotr   | Orchestral      |             |
| Romantic | Viardot,     | Pauline | Art             | Song        |
| Romantic | Warlock,     | Peter   | Art             | Song        |
| Romantic | Wolf,        | Hugo    | Art             | Song        |
| Romantic | Zumsteeg,    | Emilie  | Art             | Song        |

Table 4: The list of prompt set P in reinforcement learning experiments.