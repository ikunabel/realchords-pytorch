# Adaptive Accompaniment with ReaLchords

Yusong Wu 1 2 Tim Cooijmans <sup>2</sup> Kyle Kastner <sup>3</sup> Adam Roberts <sup>1</sup> Ian Simon <sup>1</sup> Alexander Scarlatos 1 4 Chris Donahue 1 5 Cassie Tarakajian <sup>1</sup> Shayegan Omidshafiei 6 7 Aaron Courville 2 8 Pablo Samuel Castro 1 2 Natasha Jaques 1 9 Cheng-Zhi Anna Huang 1 2 8

## Abstract

Jamming requires coordination, anticipation, and collaborative creativity between musicians. Current generative models of music produce expressive output but are not able to generate in an *online* manner, meaning simultaneously with other musicians (human or otherwise). We propose ReaLchords, an online generative model for improvising chord accompaniment to user melody. We start with an online model pretrained by maximum likelihood, and use reinforcement learning to finetune the model for online use. The finetuning objective leverages both a novel reward model that provides feedback on both harmonic and temporal coherency between melody and chord, and a divergence term that implements a novel type of distillation from a teacher model that can see the future melody. Through quantitative experiments and listening tests, we demonstrate that the resulting model adapts well to unfamiliar input and produce fitting accompaniment. ReaLchords opens the door to live jamming, as well as simultaneous co-creation in other modalities.

## 1. Introduction

Deep generative models produce realistic, high-quality content, and are seeing increasing integration into the creative processes of artists. However, such models tend not to be designed for the demands of live scenarios such as interactive improvisation, which requires anticipation of others' intentions and adaptation to mistakes, stylistic choices and deliberate exploration. We focus on music in particular, which is inherently interactive and dynamic and revolves around anticipatory collaboration in ensemble settings. Most existing models in this space, while capable of creating expressive compositions or accompaniments, are not suited for simultaneous creation, where adaptation to and alignment with the ongoing musical structure are crucial.

This paper introduces ReaLchords, a generative model tailored for online adaptive musical accompaniment. Emulating the spontaneity of live music jamming, ReaLchords generates chord accompaniments in response to a stream of monophonic melody notes, adapting on-the-fly to the unfolding musical narrative. Each chord must be generated without knowing in advance which melody note it will accompany. This simultaneous interaction imposes a conditional independence assumption on the joint generative process, that an *online* model must respect. Moreover, a model must be able to gracefully handle unfamiliar situations and unexpected changes. Likelihood models, however, suffer from exposure bias due to being trained entirely on ground truth data, and transfer poorly to the online settings where mistakes, imperfections and stylistic differences are common (see Figure [1](#page-1-0) for an example).

To address this, we use RL finetuning to improve the model with respect to reward models that consider musical coherence ([§3.2\)](#page-3-0). These reward models see the entire composition and evaluate its musical coherence from various perspectives ([§3.3\)](#page-4-0). Our setup bears similarities to RLHF [\(Ouyang](#page-9-0) [et al.,](#page-9-0) [2022;](#page-9-0) [Jaques et al.,](#page-9-1) [2019\)](#page-9-1) and RLAIF [\(Bai et al.,](#page-8-0) [2022;](#page-8-0) [Lee et al.,](#page-9-2) [2023\)](#page-9-2), however our reward models are trained through self-supervision rather than human labeling. Finally, we combine RL finetuning with knowledge distillation [\(Agarwal et al.,](#page-8-1) [2023;](#page-8-1) [Zhou et al.,](#page-10-0) [2023\)](#page-10-0) in a novel way, distilling from a teacher that can see the future into a student that cannot, hence forcing anticipation ([§3.4\)](#page-4-1).

We develop key algorithmic components (Figure [2\)](#page-3-1) needed to produce an online adaptive accompaniment model that is amenable to interactive use. Our contributions and findings are as follows: [<sup>1</sup>](#page-0-0)

<sup>1</sup>Google DeepMind <sup>2</sup>Mila - Quebec AI Institute, Universite de Montr ´ eal ´ <sup>3</sup>Google <sup>4</sup>University of Massachusetts Amherst <sup>5</sup>Carnegie Mellon University <sup>6</sup>Work done while at Google <sup>7</sup> Field AI <sup>8</sup>Canada CIFAR AI Chair <sup>9</sup>University of Washington. Correspondence to: Yusong Wu <wu.yusong@mila.quebec>, Cheng-Zhi Anna Huang <anna.huang@mila.quebec>.

<span id="page-0-0"></span><sup>1</sup>Listen to audio examples here: [https://storage.](https://storage.googleapis.com/realchords/index.html) [googleapis.com/realchords/index.html](https://storage.googleapis.com/realchords/index.html).

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1. Online models finetuned with RL are able to recover from mistakes, while models trained with MLE alone do not. We take a melody from the test set and midway introduce an abrupt transposition designed to disrupt the accompaniment model (top row). The Online MLE model predicts a bad chord (B7) and fails to adapt. ReaLchords also predicts a bad chord (F♯m), but adapts quickly. Wrong chords highlighted in orange are our own judgment informed by music theory, but the overall pattern is corroborated by an objective measure of harmonic quality, averaged over many trials of this experiment (bottom row).

♯ We propose ReaLchords, an online accompaniment generation model trained by RL finetuning. Figure [1](#page-1-0) shows how ReaLchords adapts to out-ofdistribution input, a necessary skill for live jamming. ♯ We leverage knowledge distillation to learn from a non-causal teacher that can see the future ([§3.4\)](#page-4-1). Distillation greatly improves the quality of the model, as evidenced by the human evaluation shown in Figure [3.](#page-5-0) ♯ We further employ a novel set of self-supervised reward models to encourage musical coherence and perceptual quality ([§3.3\)](#page-4-0). Based on a human listening test, we show that our reward models align closely with human preferences (Figure [3\)](#page-5-0), despite being trained without human feedback ([§3.3\)](#page-4-0). ♯ We demonstrate through a series of controlled experiments that without RL finetuning, models fail to adapt to mistakes and perturbations (Figure [4,](#page-7-0) [§5.4\)](#page-7-1). ♯ Finally, we analyze the behavior of our models in terms of domain-specific metrics (Table [1,](#page-6-0) [§5.3\)](#page-6-1). We find that each component in our RL finetuning methods improves the rhythmic and harmonic quality of generated accompaniments.

## 2. Related Work

Adaptive music accompaniment systems In contrast to automatic music generative systems, accompaniment systems often take input (such as melody) from a user, and generate output that is meant to be played in synchrony to

complement what the user is playing. Some of these systems are asynchronous, where the user first provides the full melody, and the system generates an accompaniment offline. Examples include MySong [\(Simon et al.,](#page-9-3) [2008\)](#page-9-3), where a user sings a melody and the system generates chords to accompany them. Most recently, SingSong [\(Donahue et al.,](#page-8-2) [2023\)](#page-8-2) supports a very similar interaction, but generates fullband backing tracks. Both are offline systems.

In contrast, online accompaniment systems need to synchronize with user actions in real-time. Score-following is a special case where the system has the score, the full context of the content of what the musician will play, but still needs to follow along and infer *when* to play their own part. Music Plus One [\(Raphael,](#page-9-4) [2010\)](#page-9-4) adapts its playback speed of an orchestral recording (without the soloist) to a soloist's expressive performance. Similarly, Antescofo [\(Cont,](#page-8-3) [2008\)](#page-8-3) follows where a soloist is in a score and triggers live electronics accordingly.

Generative accompaniment systems or more generally cocreative music systems, not only have to anticipate user actions, they need to learn how to respond. Voyager [\(Lewis,](#page-9-5) [2003\)](#page-9-5) takes a rule-based approach in how to listen, respond and generate musical material on the fly, while Omax Brothers [\(Assayag et al.,](#page-8-4) [2006\)](#page-8-4) recombines what a musician plays on-the-fly as an accompaniment but often requires another computer musician to control when it comes in and what section of material to draw from. ImproteK and later DJazz [\(Nika & Chemillier,](#page-9-6) [2012;](#page-9-6) [Nika et al.,](#page-9-7) [2017\)](#page-9-7) leverages a shared predefined chord progressions (such as a Jazz Standard) to coordinate the human-machine improvisation. Instead of tight synchronization, Spire Muse [\(Thelle](#page-9-8)

[& Pasquier,](#page-9-8) [2021\)](#page-9-8) serves as a brainstorming partner which retrieves musical responses that are more or less similar depending on if the user is in a converging or diverging phase of ideation.

Recent systems based on deep neural networks have emerged. BachDuet [\(Benetatos et al.,](#page-8-5) [2020\)](#page-8-5) trains an LSTM model using MLE for counterpoint (melody to bassline) accompaniment. SongDriver [\(Wang et al.,](#page-9-9) [2022\)](#page-9-9) focuses on online melody-to-chord accompaniment, similar to our work. To address exposure bias, SongDriver employs two MLE-trained models: a transformer model that predicts current output based on both current and past outputs, and a conditional random field (CRF) model that predicts current output based on previous context. The CRF model makes online predictions but does not use its own predictions for future contexts; instead, it relies on the transformer model for context.

In contrast, our system ReaLchords learns how to respond and in tight synchronization with user melody, by first learning interdependencies between melody and accompaniment from existing songs, and then using RL to tune the models to respond in an adaptive fashion.

RL finetuning for generative models Reinforcement learning (RL) finetuning has proven effective in aligning language models with human preferences [\(Ouyang et al.,](#page-9-0) [2022;](#page-9-0) [Jaques et al.,](#page-9-1) [2019\)](#page-9-1) and constraints [\(Jaques et al.,](#page-9-10) [2017\)](#page-9-10), which are often unaddressed in generative pretraining. In some cases, RL finetuning has been applied to enhance music generation models [\(Jaques et al.,](#page-9-10) [2017;](#page-9-10) [Jiang et al.,](#page-9-11) [2020b\)](#page-9-11). Most closely related to our work is RL-Duet [\(Jiang](#page-9-11) [et al.,](#page-9-11) [2020b\)](#page-9-11), which considers a similar online generation setting, namely a duet between a user and an agent, both of them playing each note without knowing what the other will play. Our work provides several contributions over RL-Duet. First, RL-Duet is trained on Bach Chorales, a small dataset of approximately 400 songs following strict rules of counterpoint composition in the style of a particular composer. In contrast, our models are trained on the diverse Hooktheory dataset of 38,000 popular songs from a wide array of artists. To enable effective learning on this scale, we develop novel multiscale contrastive and discriminative reward models, and also propose a new knowledge distillation technique specifically geared toward the online generation setting. Finally, RL-Duet experiments are limited to the setting in which the RL model is primed with the first few ground-truth notes of the accompaniment, an unrealistic assumption for real-time collaborative jamming. As we will show in [§5.4,](#page-7-1) our methods are able to begin jamming with the user's melody within a few beats, and adapt to sudden perturbations in the key.

Our work is related to the emerging literature on Reinforcement Learning from AI Feedback (RLAIF) [\(Saleh et al.,](#page-9-12)

[2020;](#page-9-12) [Bai et al.,](#page-8-0) [2022;](#page-8-0) [Lee et al.,](#page-9-2) [2023\)](#page-9-2), which mitigates the need for extensive human labeling by utilizing an AI assistant for feedback generation. We use this strategy to finetune online music language models, using an MLE model to obtain a learning signal. Recently, [Agarwal et al.](#page-8-1) [\(2023\)](#page-8-1) have shown that adding a distillation objective between the policy and a larger teacher model during RL finetuning further improves performance. ReaLchords employs a novel knowledge distillation objective between the online policy and an offline model which can see future context, bridging the gap between online improvisational capabilities and offline musical coherence.

## 3. Online Musical Accompaniment

We seek a generative model that can be used for interactive music accompaniment, where a user plays a melody, and the model simultaneously plays chords to support it. Accompaniment is a special case of the general setting in which two agents generate a joint sequence (x1, y1), . . . ,(x<sup>T</sup> , y<sup>T</sup> ) in chronological order. At each step t, agents observe the historical material x<t, y<t, and simultaneously emit the next pair of tokens xt, yt. Simultaneity imposes a conditional independence on the generative process:

Pr(xt, y<sup>t</sup> | x<t, y<t) = Pr(x<sup>t</sup> | x<t, y<t) Pr(y<sup>t</sup> | x<t, y<t).

In this general setting, the melody x and chords y interdepend through the conditioning on shared history x<t, y<t; this corresponds to musicians adapting to each other as they play. As a first step, we consider the specific setting where the chords do not influence the melody; now one player leads and the other follows. We call this accompaniment.

We approach this problem by constructing a model π<sup>θ</sup> that generates accompaniment y according to a specific autoregressive process:

<span id="page-2-1"></span><sup>π</sup>θ(<sup>y</sup> <sup>|</sup> <sup>x</sup>) = Y t πθ(y<sup>t</sup> | x<t, y<t). (1)

While our goal at each timestep t is to predict a chord y<sup>t</sup> that supports the melody token x<sup>t</sup> about to be played, the model's prediction of y<sup>t</sup> does not depend on xt. This is crucial, as it allows the model to be used online as desired.

We train this model in two steps: pretraining on data ([§3.1\)](#page-2-0), followed by finetuning using reinforcement learning ([§3.2\)](#page-3-0). In the rest of this section, we first describe the general approach, and then detail on the components involved (reward models [§3.3,](#page-4-0) distillation [§3.4,](#page-4-1) and regularizations [§3.5\)](#page-4-2).

#### <span id="page-2-0"></span>3.1. Maximum Likelihood Pretraining

The first step in training π<sup>θ</sup> is to apply MLE, maximizing the data likelihood with respect to θ:

max θ E x,y∼pdata log πθ(y | x).

<span id="page-3-1"></span>![](_page_3_Diagram_2.jpeg)

Figure 2. ReaLchords leverages RL finetuning to learn anticipation and adaptation for online melody-to-chord accompaniment. Initializing from a model π<sup>θ</sup> pretrained by MLE, the policy generates a complete chord response to a melody from the dataset, each chord being predicted given only previous melody and chords (top left). In contrast, the offline model ϕ<sup>ω</sup> (also trained by MLE) predicts each chord given the complete melody (bottom left). A KL-divergence penalty distills the predictions of the offline model into the online model, improving its ability to anticipate the future. (Right) The reward stems from an ensemble of multi-scale contrastive and discriminative models that evaluate the musical coherence between melody and chord. The final training objective in ReaLchords is a sum of the reward and the distillation loss (center).

The data distribution pdata can be interpreted as standing in for puser: we simulate user play by sampling fixed melodies from the dataset. This limits our ability to encourage and assess the model's ability to adapt to out-of-distribution melodies. Nevertheless, the model will still encounter outof-distribution combinations of melodies and chords during inference.

Unfortunately, applying only MLE training to online accompaniment model suffers from exposure bias [\(Arora](#page-8-6) [et al.,](#page-8-6) [2022\)](#page-8-6): during training, the model is always conditioned on ground-truth context, but this does not occur during inference. Consequently, MLE models struggle to learn two skills required in online accompaniment [\(Jiang](#page-9-11) [et al.,](#page-9-11) [2020b;](#page-9-11)[a\)](#page-9-13). First, the model must *anticipate* what the user is going to play, in order to ensure that its own output agrees with that of the user. Second, the model must be able to *adapt* to and recover from unknown input, due to its own mistakes or those of the user, due to misanticipation, or due to user idiosyncrasies.

As a concrete example, Figure [1](#page-1-0) shows a failure mode of the online MLE model. The model fails to adequately anticipate future inputs, leading to exposure bias and error accumulation due to a distribution mismatch between training and inference. Whenever the first few time-steps of output do not fit with the melody input stream the model will continue its chord progression, *ignoring the input*.

#### <span id="page-3-0"></span>3.2. Finetuning using Reinforcement Learning

Similar challenges are encountered in imitation learning [\(Ross & Bagnell,](#page-9-14) [2010\)](#page-9-14), where policies trained by MLE to reproduce expert demonstrations are brittle, and fail to transfer to the real environment (see e.g. [Reddy et al.](#page-9-15) [\(2019\)](#page-9-15)). A rich history of work has demonstrated Reinforcement Learning (RL) finetuning to be an effective remedy.

We begin by initializing the weights of our RL policy π<sup>θ</sup> with those of the pretrained online MLE model. As in eq. [1,](#page-2-1) at timestep t, the policy predicts action probability distribution a<sup>t</sup> = y<sup>t</sup> given state s<sup>t</sup> = (x<t, y<t). Then, we adopt an RL finetuning methodology similar to the popular RLHF (RL from Human Feedback) framework used for language models [\(Ouyang et al.,](#page-9-0) [2022;](#page-9-0) [Jaques et al.,](#page-9-1) [2019\)](#page-9-1). Namely, in addition to maximizing RL rewards R(x, y), we minimize KL-divergence from a pretrained MLE anchor model ϕω(y|x) parameterized by ω, as proposed in [Jaques](#page-9-10) [et al.](#page-9-10) [\(2017\)](#page-9-10). Let x and y represent the full melody and chord sequence, each consisting of several tokens (i.e. the full *trajectory*). This gives us the KL-regularized RL objective:

<span id="page-3-2"></span>max θ E x∼pdata y∼πθ(·|x) R(x, y) − βDKL(πθ( · | x) ∥ ϕω( · | x)). (2)

To evaluate [\(2\)](#page-3-2), we sample a batch of melodies x from the training set, then use the current policy π<sup>θ</sup> according to [\(1\)](#page-2-1) to generate a batch of corresponding harmonies y (Figure [2,](#page-3-1) top left). We then evaluate the resulting batch of compositions (x, y) according to reward models ([§3.3\)](#page-4-0) and

regularizers ([§3.5\)](#page-4-2) to obtain the reward R(x, y) (Figure [2,](#page-3-1) top and bottom right). Additionally, we measure ϕω(y | x) under the offline model ϕ<sup>ω</sup> ([§3.4\)](#page-4-1) in order to compute the KL term (Figure [2,](#page-3-1) bottom left). Finally, we update the model according to [\(2\)](#page-3-2), using REINFORCE with a separate value model serves as baseline estimation for improved stability [\(Lee et al.,](#page-9-2) [2023;](#page-9-2) [Agarwal et al.,](#page-8-1) [2023\)](#page-8-1). The separate value model is also initialized from pretrained online MLE model, and is trained to estimate the total return. We use mean square error between the estimated return and total return as objective to train the value model.

Unlike in RLHF [\(Ouyang et al.,](#page-9-0) [2022\)](#page-9-0) and RLAIF [\(Bai](#page-8-0) [et al.,](#page-8-0) [2022\)](#page-8-0), our reward models are not trained on preference labels from either human or machine labelers. Instead, they are trained using positive and negative melody-chord pairs constructed from a dataset (see Figure [2,](#page-3-1) [§3.3\)](#page-4-0). Nevertheless, a listening test ([§5.1\)](#page-5-1) shows that our reward models align well with human preferences, as shown in Figure [3.](#page-5-0)

#### <span id="page-4-0"></span>3.3. Reward Models

We develop a novel ensemble of reward models that evaluates the coherence between input (melody) and output (chord) tracks. We implement two types of coherence evaluation reward models, contrastive and discriminative, each with different inductive biases. Reward model training and architectural details can be found in Appendix [§F](#page-12-0) and [§G.](#page-13-0)

The contrastive reward model consists of a melody encoder and a chord encoder, which respectively map the melody x and chord y to embedding vectors Ex, Ey. The encoders are trained in an unsupervised manner using InfoNCE loss [\(Oord et al.,](#page-9-16) [2018;](#page-9-16) [Radford et al.,](#page-9-17) [2021\)](#page-9-17) applied to positive and negative samples created from the dataset. As shown in Figure [2,](#page-3-1) the positive pairs are defined as the melody-chord pairs from the same song, and the negative pairs are created by randomly pairing melody and chord from different songs. The InfoNCE loss essentially maximizes the cosine similarity for positive pairs, and minimizes the cosine similarity for negative pairs. The reward for a given pair x, y is the cosine similarity of E<sup>x</sup> and Ey.

The discriminative reward model looks at the entire generated pair (x, y) as a whole. This model is trained in an unsupervised manner to discriminate between "real" melodychord pairs and randomly paired melodies and chords. Each training batch case provides a set of positive pair and, by combining its melody with the chords from another randomly chosen batch case, a negative pair. Once trained, the model produces a probability of (x, y) being "real" that directly serves as the reward.

Due to the bottleneck on the embedding vectors Ex, Ey, the contrastive models focus on global coherence. The discriminative models on the other hand are able to evaluate

temporally local coherence. Indeed, our experiments in [§5.3](#page-6-1) show that contrastive reward models promote mainly harmonic quality whereas discriminative reward models encourage mainly synchronization.

While these reward models are effective, we find that they can be overly harsh on temporally localized incompatibilities, such as short-lived mistakes that are quickly resolved. To mitigate this and improve temporal credit assignment, we further propose to use an ensemble of *multi-scale* variants that evaluate isolated fragments without being influenced by distant mistakes. We train multiple contrastive and discriminative reward models to judge fragments of reduced lengths ({ 1 , 1 4 , 1 8 , 1 <sup>16</sup> } of the maximum sequence length 256). During finetuning, we apply these models to sliding windows (50% overlap) of the example.

#### <span id="page-4-1"></span>3.4. Distillation from Offline Teacher Model

As stated in [§3.2,](#page-3-0) during RL finetuning we penalize KLdivergence from a model pretrained on the data distribution to ensure the model maintains realistic outputs while maximizing rewards [\(Jaques et al.,](#page-9-10) [2017\)](#page-9-10). However, unlike in typical RL finetuning, the online MLE model with which our policy is initialized suffers from a lack of robustness to out-of-distribution data, and as such is not an ideal anchor for use with the KL-regularization term. [Agarwal et al.](#page-8-1) [\(2023\)](#page-8-1) demonstrated how the KL penalty can be used not just to avoid diverging from a checkpoint, but also to distill knowledge from a larger teacher model. We take this idea one step further and distill knowledge from an *offline* model that can see the future of the melody.

The offline model ϕ is trained with MLE to autoregressively predict chords given the full melody x:

<sup>ϕ</sup>ω(<sup>y</sup> <sup>|</sup> <sup>x</sup>) = Y t ϕω(y<sup>t</sup> | x, y<t). (3)

In traditional knowledge distillation, ground truth data is used to obtain the predictions of both the teacher and student models, and a KL loss is then applied to bring the student's predictions closer to the teacher's. Here, it is instead evaluated on samples generated by the current policy. This is a special case of *on-policy knowledge distillation* [\(Agarwal](#page-8-1) [et al.,](#page-8-1) [2023;](#page-8-1) [Zhou et al.,](#page-10-0) [2023\)](#page-10-0), which in general allows any mixture of ground truth data, student samples and teacher samples. We tested various on-policy knowledge distillation schedules and found it works best when driven by the student ([§5.3\)](#page-6-1). Thus, during RL finetuning we only train on outputs from the student.

#### <span id="page-4-2"></span>3.5. Regularization Penalties

RL finetuning can lead to pathological behavior, such as repetitions and mode collapse [\(Jaques et al.,](#page-9-10) [2017;](#page-9-10) [Jiang](#page-9-11) [et al.,](#page-9-11) [2020b\)](#page-9-11). We introduce three regularization penalties

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 3. Our reward models are aligned with human preferences. We carried out a listening test ([§5.1\)](#page-5-1) to evaluate the quality of our models. The online MLE model performs poorly, but is greatly improved by distillation from the offline MLE model. Our proposed systems ReaLchords and ReaLchords-M improve further thanks to RL finetuning. The rewards given by both the contrastive and discriminative reward models are strongly correlated with human evaluations.

to discourage specific failure modes: Repetition: Inspired by repetition penalties used for training language models (as in [Saleh et al.](#page-9-12) [\(2020\)](#page-9-12); [Jaques et al.](#page-9-1) [\(2019\)](#page-9-1)), we impose a penalty for chords that are held for too long. Silences: We impose a penalty for silences beyond the beginning of a phrase. Ending early: A penalty is imposed for early end-of-sequence (EOS) tokens. See [§D](#page-11-0) for an ablation that shows the need of these penalties.

## 4. Dataset

We train our models on an updated version of the Hooktheory dataset [\(Donahue et al.,](#page-8-7) [2022\)](#page-8-7), which comprises crowdsourced analyses of monophonic melodies and chords from recordings and now contains 38K melody-chord pairs. We adopt a frame-based representation where time is quantized to sixteenth notes, where each frame is a discrete index. We set a maximum sequence length of 256 for x and y. We augment the data by randomly transposing up or down by up to 6 semitones. 20% of the data is held out and divided equally into validation and test sets. We develop on the validation set and report the test set results in the paper. Please refer to [§L](#page-15-0) for details on the dataset and data representation.

## 5. Experiments

Is the system capable of producing accompaniments of high musical quality? How swiftly can the system adjust to unfamiliar situations? We address these questions from three directions.

To directly assess musical quality, we conduct a human listening test using samples generated from the models ([§5.1\)](#page-5-1). We demonstrate adaptation through several controlled generation experiments, tracking the quality of the accompaniment over time (explained in [§5.4\)](#page-7-1). Finally, we evaluate the system using heuristic metrics to assess the quality of compositions generated in response to melodies in the test set (detailed in [§5.3\)](#page-6-1).

The following systems are compared in our experiments:

MLE baselines The Online MLE model trained to predict y<sup>t</sup> | x<t, y<t without seeing x<sup>t</sup> ([§3.1\)](#page-2-0). The Offline MLE model that sees the full input x and is used as a teacher for knowledge distillation ([§3.4\)](#page-4-1).

Our proposals These models are trained with both contrastive and discriminative rewards, as well as regularization and knowledge distillation. ReaLchords incorporates the global reward models, whereas ReaLchords-M incorporates the multi-scale variants of both reward models.

Ablations The model KD, trained with only knowledge distillation and regularization. Two models trained by MLE and then finetuned using either only Contrastive (C) reward or only Discriminative (D) reward, with regularization and KL divergence to the MLE checkpoint. A model C+D using both contrastive and discriminative reward, with regularization and KL divergence to the online MLE checkpoint.

#### <span id="page-5-1"></span>5.1. Human and Machine Evaluation on Musicality

Any measure of the quality of a musical piece must ultimately be grounded in human preferences. We carry out a listening test to evaluate four systems: the Online MLE baseline, KD, ReaLchords and ReaLchords-M. In the listening test, participants are presented with 8-second audio clips from two different systems, and asked to rate which one sounded more musical, on a 5-point Likert scale. We recruited ten musicians, and collected 192 ratings with each system involved in 96 pairwise comparisons (see [§J](#page-15-1) for more details).

Figure [3](#page-5-0) (top) shows the number of wins for each sys-

<span id="page-6-0"></span>Table 1. Effect of RL finetuning with our reward models and knowledge distillation on harmonic, synchronization and rhythmic diversity metrics. Each number is an average over a large number of accompaniments to test set melodies. In row 2-7, we report confidence interval (95%) of metric values over 3 RL finetunings, each with different random seeds.

|                | Models       | Harmony ∆ Note in Chord ↑ , % | Synchronization Chord-Note Onset Interval ↓ , × 10 | Rhythm Diversity − 3 Chord Length Entropy ↑ |
|----------------|--------------|-------------------------------|----------------------------------------------------|---------------------------------------------|
| Online         | MLE          | 36.99                         | 14.23                                              | 2.21                                        |
| Knowledge      | Distillation | (KD) 46.38 ± 0.45             | 14.39 ± 1.25                                       | 1.80 ± 0.04                                 |
| Contrastive    | (C)          | 47.22 ± 1.22                  | 16.61 ± 4.78                                       | 1.80 ± 0.29                                 |
| Discriminative | (D)          | 44.29 ± 0.99                  | 12.34 ± 7.95                                       | 1.70 ± 0.06                                 |
| C +            | D            | 46.12 ± 0.95                  | 12.86 ± 6.17                                       | 1.56 ± 0.05                                 |
| ReaLchords     |              | 48.17 ± 0.27                  | 16.09 ± 3.63                                       | 1.35 ± 0.30                                 |
| ReaLchords     | -M           | 54.29 ± 1.55                  | 17.17 ± 4.86                                       | 1.66 ± 0.20                                 |
| Offline        | MLE          | 63.73                         | 9.85                                               | 1.90                                        |
| Test           | set          | 70.94                         | 0.00                                               | 2.19                                        |

tem. We ran a Kruskal-Wallis H test and confirmed that there are statistically significant pairs among the permutations. According to a post-hoc analysis using the Wilcoxon signed-rank test with Bonferroni correction (with p<0.05/6 as there are 6 pairs of systems), we found the following statistically significant results: All systems outperformed the Online MLE baseline. Also, the fully-fledged systems ReaLchords and ReaLchords-M outperformed distillation alone (KD). While ReaLchords-M appears to outperform ReaLchords, this comparison is not significant.

Overall, the results from the listening test show that distillation alone (KD) accounts for a large improvement in perceptual quality. The reward models agree with this assessment, even though KD does not directly optimize for these rewards. In general, we find that the rewards given by our self-supervised reward models (Figure [3,](#page-5-0) middle and bottom) correlate strongly with human preferences, which justifies their use in lieu of human feedback.

#### 5.2. Quantitative Metrics

In line with prior research [\(Jiang et al.,](#page-9-11) [2020b;](#page-9-11) [Yang &](#page-10-1) [Lerch,](#page-10-1) [2020;](#page-10-1) [Fang et al.,](#page-9-18) [2020\)](#page-9-18), we introduce quantitative metrics to evaluate the quality of accompaniments:

Harmonic quality We measure harmonic quality by the note-in-chord ratio, which is the amount of time that the melody's pitch class occurs in the chord. For example, if the melody token x<sup>t</sup> is a C, and the chord y<sup>t</sup> is F minor, then the note-in-chord ratio as time t equals 1. We average this metric across time t and across all compositions x, y generated, to obtain the overall note-in-chord ratio for the model in question.

Synchronization To gauge temporal synchronization between melody and chord progression, we look at *chord-tonote onset interval*, which is the length of time between

the onset of a chord and the onset of the nearest preceding melody note. The synchronization of a model can be judged by comparing this quantity's distribution on the test set versus on the output of the model. Whereas [Jiang et al.](#page-9-11) [\(2020b\)](#page-9-11) compare averages of this quantity, we propose to compare the full distributions using Earth Mover's Distance (EMD) on histograms of chord-to-note onset intervals.

Rhythmic diversity We examine the distribution of durations of generated chords to assess overall rhythmic behavior. The entropy of this distribution measures rhythmic diversity.

#### <span id="page-6-1"></span>5.3. Quantitative Evaluation Results

We evaluate each model based on a large number of accompaniments to test set melodies. The average metrics are reported in Table [1.](#page-6-0)

MLE baselines The behavior of Offline MLE is closest to that of the test set, as is expected due to its ability to see the future input. Online MLE exhibits poor harmonic and temporal coordination with the melody, which suggests that it produces chords without paying attention to the melody.

Distillation On-policy knowledge distillation (KD in Table [1\)](#page-6-0) significantly enhances online generation, particularly with regard to harmony and synchronization. Distillation from the offline teacher will suppress the probability of chords that match poorly with the future melody, forcing the student to anticipate the immediate future. The student (KD) learns to produce outputs that align better with the input context while retaining a causal conditioning structure.

Reward models Training with contrastive and discriminative reward models, individually (C, D) and combined (C + D), shows distinct improvements in harmony and synchronization. The use of the contrastive reward model (C) improves more on harmony, presumably because it compresses the entire melody and the entire accompaniment separately

<span id="page-7-0"></span>(a) Accompaniment quality when primed with ground truth.

![](_page_7_Figure_3.jpeg)

(b) Accompaniment quality after a cold start.

![](_page_7_Figure_6.jpeg)

(c) Accompaniment quality when perturbed midway.

![](_page_7_Figure_10.jpeg)

Figure 4. Comparing the quality of overall accompaniment as a function of the number of beats generated, in three scenarios of increasing difficulty ([§5.4\)](#page-7-1). Quality is measured by note-in-chord ratio. (a) Priming the online model with ground-truth context (8 beats in this case) results in comparable performance between models. (b-c) ReaLchords and ReaLchords-M recover from cold starts and perturbation, while the online model does not.

and merges them only in the final cosine similarity. The use of the discriminative reward model (D) improves more on synchronization, while having worse harmony. This is expected, as the classification is biased towards direct comparison. We further examine the bias of different reward models by plotting reward values against harmonic perturbations, where varying portions of chords in the test set are replaced with random alternatives. As shown in Figure [6](#page-13-1) in [§G,](#page-13-0) the contrastive model is more sensitive to harmonic perturbations.

The blend of both rewards in ReaLchords offers enhancement in each metrics. Additional experiments applying RL fine-tuning with ensembles of the same type of reward models show similar metric improvements, as presented in Table [6](#page-14-0) in [§I.](#page-14-1) This suggests that the observed metric enhancements may result from both the combined biases and the ensemble of reward models.

Combining rewards with distillation Integrating both reward models with knowledge distillation yields better harmony but less rhythmic diversity (ReaLchords in Ta-

ble [1\)](#page-6-0). This indicates a tendency of the model to opt for 'safer' chord progressions, presumably resulting from satisfying both reward maximization and knowledge distillation. This is further validated in Figure [8](#page-16-0) in Appendix [§M](#page-15-2) where we visualize the chord length histograms and find that this model tends to hold chords for 2 or 4 beats.

Multi-scale reward models ReaLchords-M further improves harmonic quality thanks to the locally isolated rewards from the multi-scale variants of our reward models. This aligns well with the findings in Figure [3.](#page-5-0)

#### <span id="page-7-1"></span>5.4. Adaptation Dynamics

To study the temporal dynamics of model adaptation to unknown input, we measure accompaniment quality as a function of number of beats generated. A beat is 4 frames with a total of a quarter note length. We compare among the dataset and four models: Online MLE, Offline MLE, ReaLchords and ReaLchords-M. We report harmonic quality in terms of note-in-chord ratio. In all experiments, we draw melodies from the test set, and let the models generate accompaniment. However, we consider three different scenarios with different interventions: priming, cold-start, and perturbation.

Priming (Figure [4a\)](#page-7-0): We start with the setting of RL-Duet [\(Jiang et al.,](#page-9-11) [2020b\)](#page-9-11), where the models are primed with several beats of ground truth chords before generating their own chords. This avoids the cold-start problem of predicting a chord without knowing anything about the melody, and gives an indication of the model's ability to anticipate what happens next without having to first adapt to what happened previously. Behavior is similar across the models. For reference, we also plot the cold-start behavior of Online MLE (without priming), which is significantly worse. We argue that, as a benchmark for online accompaniment, primed generation is unnatural and too facile.

Cold start (Figure [4b\)](#page-7-0): We now proceed to the cold-start setting, which is more natural and more difficult. Here, models predict chords immediately and have to adapt to the resulting melody-chord combinations, which are usually wrong and outside of the data distribution. The Online MLEstruggles to adapt to its own past chords, and never gets close to the primed behavior. ReaLchords and ReaLchords-M quickly overcome their own mistakes and play as well as if they were primed.

Perturbation (Figure [4c\)](#page-7-0): Finally, we introduce a deliberate perturbation in the middle of the generation process, to demonstrate the ability of our systems to recover from serious errors. We transpose the melody up by a tritone (6 semitones) at beat 17, resulting in both an out-of-distribution melody and almost guaranteeing that the next chord is a poor fit. This is similar to the push test in legged robotics. The Online MLE fails the test: it exhibits a drop in harmonic quality and never recovers. ReaLchords and ReaLchords-M quickly adapt to the new key and recover their previous performance.

Overall, these results confirm that Online MLE suffers from exposure bias due to only being trained on ground-truth data. This brittleness, or inability to produce reasonable output given out-of-distribution (OOD) input not covered by the training data, is similar to the failures exhibited by imitation learning or behavior cloning methods in traditional RL contexts [\(Reddy et al.,](#page-9-15) [2019\)](#page-9-15), which also rely purely on supervised learning. Our systems ReaLchords and ReaLchords-M quickly recover from both cold-start situations and mid-song disturbances, which highlights their ability to follow along with a user as they explore ideas. Thus, ReaLchords solves a critical requirement for an online accompaniment system, which will have to accompany a diverse range of human users who are likely to play novel melodies not covered by the training data and change what they play midway.

Finally, we find an interesting emergent behavior due to RL finetuning, where models hold off on playing chords initially, preferring instead to wait for more information about the melody. This *wait and see* behavior is also visible in Figure [1,](#page-1-0) and examined further in Appendix [§A.](#page-11-1) Similar behavior occurs in human performers, who often wait for several bars when improvising with an unfamiliar player.

## 6. Conclusion

We proposed ReaLchords, an online generative model that improvise simultaneous chord accompaniment in response to melody input. ReaLchords leverages RL finetuning with multi-scale contrastive and discriminative reward models and employs a novel offline-to-online knowledge distillation technique. Throughout our experiments, we show that ReaLchords accompanies with good harmony and synchronization, while effectively adapting to mistakes and perturbations. We also show that listeners preferred ReaLchords over the online MLE baseline and distillation-only models, while validating the proposed reward models align with human judgement. ReaLchords enables an exciting path towards an effective and engaging real-time, interactive music accompaniment system.

# <span id="page-8-3"></span>Acknowledgement

<span id="page-8-7"></span><span id="page-8-2"></span>We would like to express our gratitude to the following individuals for their insightful discussions and valuable advice on the project: Jesse Engel, Ethan Manilow, Antoine Caillon, Laura Graesser, Athul Jacob, Kory Mathewson, Max Schwarzer, Evgenii Nikishin, Zhixuan Lin, Dinghuai Zhang, Michael Noukhovitch, Ke Chen, Yuxuan Wu, Yi

Deng. We also like to thank the individuals who designed and built the RL training infrastructure used in this paper: Leonard Hussenot, Johan Ferret, Robert Dadashi, Geoffrey ´ Cideron, Alexis Jacq, Sabela Ramos, Piotr Stanczyk, Sertan Girgin, Danila Sinopalnikov, Amelie H ´ eliou, Bobak ´ Shahriari, Bilal Piot, Matt Hoffmann, Nikola Momchev, and Olivier Bachem.

# Impact Statement

While this work is developed in the domain of music, the interactive dynamic of simultaneous musical accompaniment has broader societal implications beyond the realm of music generation. This research could also influence the way AI is perceived in creative fields, reinforcing the potential of AI as a collaborative tool rather than a replacement for human creativity.

## References

<span id="page-8-6"></span><span id="page-8-5"></span><span id="page-8-4"></span><span id="page-8-1"></span><span id="page-8-0"></span>Agarwal, R., Vieillard, N., Stanczyk, P., Ramos, S., Geist, M., and Bachem, O. GKD: Generalized knowledge distillation for auto-regressive sequence models. *arXiv preprint arXiv:2306.13649*, 2023. Arora, K., Asri, L. E., Bahuleyan, H., and Cheung, J. C. K. Why exposure bias matters: An imitation learning perspective of error accumulation in language generation. *arXiv preprint arXiv:2204.01171*, 2022. Assayag, G., Bloch, G., Chemillier, M., Cont, A., and Dubnov, S. Omax brothers: a dynamic yopology of agents for improvization learning. In *Proceedings of the 1st ACM workshop on Audio and music computing multimedia*, 2006. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. Constitutional ai: Harmlessness from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022. Benetatos, C., VanderStel, J., and Duan, Z. Bachduet: A deep learning system for human-machine counterpoint improvisation. In *Proceedings of the International Conference on New Interfaces for Musical Expression*, 2020. Cont, A. Antescofo: Anticipatory synchronization and control of interactive parameters in computer music. In *International Computer Music Conference (ICMC)*, 2008. Donahue, C., Thickstun, J., and Liang, P. Melody transcription via generative pre-training. In *Proceedings of ISMIR 2022*, 2022. Donahue, C., Caillon, A., Roberts, A., Manilow, E., Esling, P., Agostinelli, A., Verzetti, M., Simon, I., Pietquin,

<span id="page-9-23"></span><span id="page-9-22"></span><span id="page-9-21"></span><span id="page-9-20"></span><span id="page-9-19"></span><span id="page-9-18"></span><span id="page-9-17"></span><span id="page-9-16"></span><span id="page-9-15"></span><span id="page-9-14"></span><span id="page-9-13"></span><span id="page-9-12"></span><span id="page-9-11"></span><span id="page-9-10"></span><span id="page-9-9"></span><span id="page-9-8"></span><span id="page-9-7"></span><span id="page-9-6"></span><span id="page-9-5"></span><span id="page-9-4"></span><span id="page-9-3"></span><span id="page-9-2"></span><span id="page-9-1"></span><span id="page-9-0"></span>O., Zeghidour, N., et al. Singsong: Generating musical accompaniments from singing. *arXiv preprint arXiv:2301.12662*, 2023. Fang, A., Liu, A., Seetharaman, P., and Pardo, B. Bach or mock? a grading function for chorales in the style of js bach. *arXiv preprint arXiv:2006.13329*, 2020. Jaques, N., Gu, S., Bahdanau, D., Hernandez-Lobato, J. M., ´ Turner, R. E., and Eck, D. Sequence tutor: Conservative fine-tuning of sequence generation models with klcontrol. In *International Conference on Machine Learning*, pp. 1645–1654. PMLR, 2017. Jaques, N., Shen, J. H., Ghandeharioun, A., Ferguson, C., Lapedriza, A., Jones, N., Gu, S. S., and Picard, R. Humancentric dialog training via offline reinforcement learning. *Empirical Methods in Natural Language Processing (EMNLP)*, 2019. Jiang, N., Jin, S., Duan, Z., and Zhang, C. When counterpoint meets chinese folk melodies. *Advances in neural information processing systems*, 33:16258–16270, 2020a. Jiang, N., Jin, S., Duan, Z., and Zhang, C. Rl-duet: Online music accompaniment generation using deep reinforcement learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 710–718, 2020b. Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014. Lee, H., Phatale, S., Mansoor, H., Lu, K., Mesnard, T., Bishop, C., Carbune, V., and Rastogi, A. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. *arXiv preprint arXiv:2309.00267*, 2023. Lewis, G. E. Too many notes: Computers, complexity, and culture in voyager. In *New Media*, pp. 93–106. Routledge, 2003. Nika, J. and Chemillier, M. Improtek: integrating harmonic controls into improvisation in the filiation of omax. In *International Computer Music Conference (ICMC)*, pp. 180–187, 2012. Nika, J., Chemillier, M., and Assayag, G. Improtek: introducing scenarios into human-computer music improvisation. *Computers in Entertainment (CIE)*, 14(2), 2017. Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*, 2018. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pp. 8748–8763. PMLR, 2021. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, 21(1):5485–5551, 2020. Raphael, C. Music plus one and machine learning. In *ICML*, 2010. Reddy, S., Dragan, A. D., and Levine, S. Sqil: Imitation learning via reinforcement learning with sparse rewards. *arXiv preprint arXiv:1905.11108*, 2019. Roberts, A., Chung, H. W., Mishra, G., Levskaya, A., Bradbury, J., Andor, D., Narang, S., Lester, B., Gaffney, C., Mohiuddin, A., et al. Scaling up models and data with t5x and seqio. *Journal of Machine Learning Research*, 24(377):1–8, 2023. Ross, S. and Bagnell, D. Efficient reductions for imitation learning. In *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, pp. 661–668. JMLR Workshop and Conference Proceedings, 2010. Saleh, A., Jaques, N., Ghandeharioun, A., Shen, J., and Picard, R. Hierarchical reinforcement learning for opendomain dialog. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 8741–8748, 2020. Shazeer, N. and Stern, M. Adafactor: Adaptive learning rates with sublinear memory cost. In *International Conference on Machine Learning*, pp. 4596–4604. PMLR, 2018. Simon, I., Morris, D., and Basu, S. Mysong: automatic accompaniment generation for vocal melodies. In *Proceedings of the SIGCHI conference on human factors in computing systems*, 2008. Thelle, N. J. and Pasquier, P. Spire muse: A virtual musical partner for creative brainstorming. In *NIME 2021*. PubPub, 2021. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. *Advances in neural information processing systems*, 30, 2017. Wang, Z., Zhang, K., Wang, Y., Zhang, C., Liang, Q., Yu, P., Feng, Y., Liu, W., Wang, Y., Bao, Y., et al. Songdriver:

Real-time music accompaniment generation without logical latency nor exposure bias. In *Proceedings of the 30th ACM International Conference on Multimedia*, pp. 1057–1067, 2022.

<span id="page-10-1"></span>Yang, L.-C. and Lerch, A. On the evaluation of generative models in music. *Neural Computing and Applications*, 32(9):4773–4784, 2020.

<span id="page-10-0"></span>Zhou, Y., Lyu, K., Rawat, A. S., Menon, A. K., Rostamizadeh, A., Kumar, S., Kagy, J.-F., and Agarwal, R. Distillspec: Improving speculative decoding via knowledge distillation. *arXiv preprint arXiv:2310.08461*, 2023.

<span id="page-11-2"></span>![](_page_11_Figure_1.jpeg)

Figure 5. Chord silence ratio within each beat of the generation.

#### <span id="page-11-1"></span>A. Wait and See Behavior

To further investigate the *wait and see* behavior noted at the end of [§5.4,](#page-7-1) we measure ratio of chord silence during a non-silence note across each beat. Figure [5](#page-11-2) shows that the finetuned models ReaLchords and ReaLchords-M, are often silent during the first few beats. We believe this behavior results from trading off the penalty for silence and the low reward for bad guesses.

## B. Knowledge Distillation Schedules

On-policy knowledge distillation [\(Agarwal et al.,](#page-8-1) [2023;](#page-8-1) [Zhou et al.,](#page-10-0) [2023\)](#page-10-0) allows the KL to be evaluated with respect to any mixture of policy, teacher and data distribution. We compare these options and report results in Table [2.](#page-12-1) Sampling from the policy alone results in good performance, and we use this choice in all of our experiments.

## C. Details of Regularization Penalties

RL finetuning can lead to pathological behavior, such as repetitions and mode collapse [\(Jaques et al.,](#page-9-10) [2017;](#page-9-10) [Jiang et al.,](#page-9-11) [2020b\)](#page-9-11). We introduce several regularization penalties to discourage specific failure modes.

Repetition The model has a tendency to repeat itself, generating long-held chords. Inspired by repetition penalties used for training language models (as in [Saleh et al.](#page-9-12) [\(2020\)](#page-9-12); [Jaques et al.](#page-9-1) [\(2019\)](#page-9-1)), we provide a penalty of −1 for each token in a chord that is held for longer than 32 frames (8 beats), because chords longer than 8 beats rarely occur in the training set.

Silence To avoid behavior where no chords are generated, we provide a penalty of −1 for each silence token if more than 4% of frames are silent accompaniment to a non-silent input. This is the rate at which silent accompaniment occurs in the training set. We omit this penalty for the first 8 frames (first half-note) to allow early adaptation.

Ending early To discourage early end-of-sequence (EOS) tokens, we provide a penalty of −1 for each token after model outputs EOS but before the input does. The penalties are given to the whole sequence at the last token of output before EOS. An ablation experiment in [§D](#page-11-0) demonstrates the need for these penalties.

## <span id="page-11-0"></span>D. Ablation on Regularizers

To show the need for regularization penalties, we conduct an ablation test. As shown in the Table [3,](#page-12-2) removing the repetition penalty results in significantly more long chords. Removing the silence penalty would result in a long chunk of silence at the beginning of the generation, while removing the early-finish penalty would result in high ratio of samples stop early.

## E. Training and Architecture Details of Online and Offline Models

The online model π<sup>θ</sup> is implemented as an 8-layer autoregressive decoder-only transformer [\(Vaswani et al.,](#page-9-19) [2017\)](#page-9-19) with 6 heads and hidden dimension of 512. It is trained on interleaved input and output tokens y1, x1, . . . , y<sup>T</sup> , x<sup>T</sup> , providing conditional distributions of the form πθ(y<sup>t</sup> | x<t, y<t) and πθ(x<sup>t</sup> | x<t, y⩽<sup>t</sup>). To use this model *online* as per [\(1\)](#page-2-1), we alternate drawing yts from πθ(y<sup>t</sup> | x<t, y<t) and xts from the given melody, foregoing the use of the conditionals

<span id="page-12-1"></span>Table 2. Comparing different knowledge distillation schedule, generating with policy data achieves best balance between harmonic, synchronization and rhythmic diversity metrics.

|         | Models             | ∆ Melody Note in Chord Ratio × 10 − 3 (harmony) ↑ | Chord-Note Onset Interval Chord (synchronization) ↓ (rhythm | Length Entropy diversity) ↑ |
|---------|--------------------|---------------------------------------------------|-------------------------------------------------------------|-----------------------------|
| Online  | MLE                | 36.99                                             | 14.23                                                       | 2.21                        |
| Policy  | (KD)               | 46.54                                             | 12.81                                                       | 1.79                        |
| Dataset |                    | 35.25                                             | 19.58                                                       | 2.05                        |
| Dataset | + Policy           | 46.28                                             | 13.65                                                       | 1.67                        |
| Dataset | + Policy + Teacher | 46.64                                             | 13.39                                                       | 1.63                        |
| Offline | MLE                | 63.73                                             | 9.85                                                        | 1.90                        |
| Test    | set                | 70.94                                             | 0                                                           | 2.19                        |

<span id="page-12-2"></span>Table 3. Comparing quantitative metrics of ablation experiments on regularization penalties, the proposed penalties effectively prevent RL training to exploit reward which result in sub-optimal solution. Details in [§D.](#page-11-0)

|               | Models                        | Note-in-Chord Ratio (%) | Chord-Note Onset Interval | Chord Length Entropy | Chord Silence Ratio (%) | Long Chords Ratio (%) | Early Stop Ratio (%) |
|---------------|-------------------------------|-------------------------|---------------------------|----------------------|-------------------------|-----------------------|----------------------|
| Online        | MLE                           | 36.99                   | 14.23                     | 2.21                 | 7.31                    | 1.67                  | 21.66                |
| Offline       | MLE                           | 63.73                   | 9.85                      | 1.90                 | 2.91                    | 0.73                  | 2.62                 |
| Test          | set                           | 70.94                   | 0                         | 2.19                 | 2.83                    | 0.71                  | 0                    |
| KD            |                               | 46.54                   | 12.8                      | 11.79                | 4.55                    | 1.51                  | 0                    |
| ReaLchords KD |                               | 48.17                   | 13.01                     | 1.25                 | 4.90                    | 0.02                  | 0                    |
| w/o           | all penalties ReaLchords      | 45.96                   | 14.98                     | 1.66                 | 32.67                   | 0                     | 0.38                 |
| w/o           | repetition penalty ReaLchords | 48.19                   | 13.17                     | 1.31                 | 4.81                    | 0.78                  | 0                    |
| w/o           | repetition penalty            |                         |                           |                      |                         |                       |                      |
| w/o           | silence penalty               |                         |                           |                      |                         |                       |                      |
|               | ReaLchords                    | 46.38                   | 11.12                     | 1.43                 | 26.50                   | 0.32                  | 0                    |
| w/o           | all penalties                 | 46.92                   | 11.55                     | 1.46                 | 27.85                   | 0.29                  | 0                    |

πθ(x<sup>t</sup> | x<t, y⩽<sup>t</sup>). The online model is trained using Adafactor optimizer [\(Shazeer & Stern,](#page-9-20) [2018\)](#page-9-20) and learning rate of 10−<sup>3</sup> with a batch size of 256. The online model is trained for 50, 000 steps with 1000 steps of warmup. We apply a dropout with rate 0.1 to the online model during training.

The offline model ϕ<sup>ω</sup> is implemented as an 8-layer encoder-decoder transformer in the style of T5 [\(Raffel et al.,](#page-9-21) [2020\)](#page-9-21) with 6 heads and hidden dimension of 512. This model is trained on pairs of sequences x, y without interleaving, by first encoding the entire melody x and then generating the entire chord progression y | x conditioned on the encoding of x. The offline model is trained using Adafactor optimizer [\(Shazeer & Stern,](#page-9-20) [2018\)](#page-9-20) and learning rate of 10−<sup>3</sup> with a batch size of 256. The offline model is trained for 50, 000 steps with 1000 steps of warmup. We apply a dropout with rate 0.1 to the offline model during training. We train our online and offline transformers using the T5X framework [\(Roberts et al.,](#page-9-22) [2023\)](#page-9-22).

# <span id="page-12-0"></span>F. Training and Architecture Details of Reward Models

The training specification is identical for reward models at all scale, with the only difference lies in their input length. Our contrastive reward model consists of two identical 6-layer, 6-head transformer encoders with 512 hidden dimension, one for encoding the entire melody x and one for encoding the entire chord progression y. The contrastive reward model is trained using Adam optimizer [\(Kingma & Ba,](#page-9-23) [2014\)](#page-9-23) and learning rate of 10−<sup>4</sup> with a batch size of 128. The contrastive reward model is trained for 35 epochs with a dropout rate of 0.1.

The discriminative reward model consists of a 6-layer, 6-head transformer encoder with 512 hidden dimension that encodes the entire composition x, y.The discriminative reward model is trained using Adam optimizer [\(Kingma & Ba,](#page-9-23) [2014\)](#page-9-23) and learning rate of 10−<sup>4</sup> with a batch size of 64. The discriminative reward model is trained for 5 epochs which we find after which point the model starts to overfit. We also applied a dropout rate of 0.1 during training.

#### <span id="page-13-0"></span>G. Test Set performance of Reward Models

We evaluate the contrastive and discriminative reward models on test set, and report performance in Table [4](#page-14-2) for contrastive model, and Table [5](#page-14-3) for discriminative model. For contrastive model, we evaluate it on retrieval metrics, namely recall (R@1, R@5, R@10) and mean average precision (mAP@10). For discriminative model, we use test set pair as positive label, and generate same number of negative samples by taking randomly pairing across test set. While performance of discriminative model remains similar across all scale, for contrastive model, the retrieval performance degrades at smaller scale. Given the focus of contrastive models on harmony evaluation, we posit that the observed decline in performance at smaller scales is attributed to the increased chance of harmonic coherence between unmatched samples in shorter segments.

<span id="page-13-1"></span>We examine the bias of different reward models by plotting reward values against harmonic perturbations, where varying portions of chords in the test set are replaced with random alternatives. As shown in Figure [6,](#page-13-1) the contrastive model is more sensitive to harmonic perturbations.

![](_page_13_Figure_6.jpeg)

Figure 6. We take ground-truth samples from the test set, and replace a portion of chords with random chord names but keep the chord boundary. We then run the perturbed sample through contrastive and discriminative reward models. The points in line show the average reward value while shaded area shows the confidence interval (95%).The results show that the contrastive model is more attentive to harmonic perturbation or "mistakes". Both reward models lower their scores as perturbation or "mistakes" increases.

## H. Training Details of RL Finetuning

In RL training, we use a learning rate of 10−<sup>4</sup> for both policy model and value model. In the initialization of RL finetuning, the policy model and the value model are both initialized from online MLE model. The coefficient β between reward maximization and KL loss in Equation [2](#page-3-2) is fixed as 0.5 for all the experiments. We apply a coefficient of 50 to the reward produced by reward models. We apply a coefficient of 20 to the ending early penalty in all experiments used this penalty. In experiment using only knowledge distillation objective or using only one reward model, we apply a coefficient of 1 to both the repetition penalty and silence penalty. To train ReaLchords, we apply a coefficient of 2 to both the repetition penalty and silence penalty. To train ReaLchords-M, we apply a coefficient of 10 to both the repetition penalty and silence penalty.

We do not backpropagate gradients through the sampling process of the policy. All rewards, including reward models and penalties are summed and applied to the last token policy generated. For knowledge distillation objective, we only backpropagate gradients through the tokens policy generated, excluding the input tokens.

Table 4. The retrieval performance of contrastive reward model on test set.

<span id="page-14-2"></span>

|      | Contrastive |      | Reward Models | R@1  | R@5  | Note to Chord R@10 | mAP@10 | R@1  | R@5  | Chord to Note R@10 | mAP@10 |
|------|-------------|------|---------------|------|------|--------------------|--------|------|------|--------------------|--------|
| Full | context     | (256 | frames)       | 0.17 | 0.39 | 0.49               | 0.26   | 0.17 | 0.39 | 0.51               | 0.27   |
| 1/2  | context     | (128 | frames)       | 0.05 | 0.14 | 0.21               | 0.09   | 0.05 | 0.15 | 0.21               | 0.09   |
| 1/4  | context     | (64  | frames)       | 0.02 | 0.08 | 0.13               | 0.05   | 0.02 | 0.07 | 0.12               | 0.05   |
| 1/8  | context     | (32  | frames)       | 0.02 | 0.06 | 0.10               | 0.04   | 0.02 | 0.05 | 0.09               | 0.03   |
| 1/16 | context     | (16  | frames)       | 0.01 | 0.04 | 0.07               | 0.03   | 0.01 | 0.03 | 0.06               | 0.02   |

<span id="page-14-3"></span>Table 5. The classification performance of discriminative reward model on test set.

|   | Full | Discriminative context |      | Reward (256 frames) | Models Precision 0.69 | Recall 0.91 | F1 0.79 |
|---|------|------------------------|------|---------------------|-----------------------|-------------|---------|
| 1 | / 2  | context                | (128 | frames)             | 0.69                  | 0.92        | 0.79    |
| 1 | / 4  | context                | (64  | frames)             | 0.71                  | 0.84        | 0.77    |
| 1 | / 8  | context                | (32  | frames)             | 0.69                  | 0.88        | 0.77    |
| 1 | / 16 | context                | (16  | frames)             | 0.68                  | 0.79        | 0.73    |

For systems using only reward models (C, D, C+D in Table [1\)](#page-6-0), in addition to regularization penalties, we also include the knowledge distillation loss between policy and online MLE model. Similar to [Jaques et al.](#page-9-10) [\(2017\)](#page-9-10), we found without such KL regularization, the training will be unstable and the models will generate invalid token sequences (e.g. a hold token before any onset token).

We also attempt to reproduce similar methods used in previous work RL-Duet [\(Jiang et al.,](#page-9-11) [2020b\)](#page-9-11) but received little success. We explored using offline MLE model as reward model, or using both offline MLE model and online MLE model as reward models, alongside with all regularization penalties we use. In all cases, the trained policy only generates repeated same chord, failing to generate adequate accompaniment.

# <span id="page-14-1"></span>I. Ensemble Same Type of Reward Models

We conduct experiments on RL finetuning with ensemble of same type of reward models, with results shown in Table [6.](#page-14-0) For both contrastive and discriminative model, ensemble with same kind helps improving better metric value.

<span id="page-14-0"></span>Table 6. Effect of RL finetuning with our reward models and knowledge distillation on harmonic, synchronization and rhythmic diversity metrics. In row 2-7, we report confidence interval (95%) of metric values over 3 trainings, each with different random seeds. We additionally include results of RL finetuning with C+C and D+D (two contrastive or discriminative models trained with different initial seeds). From the metrics value, the ensemble of reward models helps achieving better metric performances.

|                | Models       | Note       | Harmony in | ∆ Chord ↑ , % | Chord-Note | Synchronization Onset | Interval ↓ , × 10 | Rhythm − 3 Chord | Length | Diversity Entropy ↑ |
|----------------|--------------|------------|------------|---------------|------------|-----------------------|-------------------|------------------|--------|---------------------|
| Online         | MLE          |            | 36.99      |               |            | 14.23                 |                   |                  | 2.21   |                     |
| Knowledge      | Distillation | (KD) 46.38 | ±          | 0.45          | 14.39      | ±                     | 1.25              | 1.80             | ±      | 0.04                |
| Contrastive    | (C)          | 47.22      | ±          | 1.22          | 16.61      | ±                     | 4.78              | 1.80             | ±      | 0.29                |
| Discriminative | (D)          | 44.29      | ±          | 0.99          | 12.34      | ±                     | 7.95              | 1.70             | ±      | 0.06                |
| C              | + D          | 46.12      | ±          | 0.95          | 12.86      | ±                     | 6.17              | 1.56             | ±      | 0.05                |
| ReaLchords     |              | 48.17      | ±          | 0.27          | 16.09      | ±                     | 3.63              | 1.35             | ±      | 0.30                |
| ReaLchords     | -M           | 54.29      | ±          | 1.55          | 17.17      | ±                     | 4.86              | 1.66             | ±      | 0.20                |
| C              | + C          | 47.30      | ±          | 1.63          | 10.13      | ±                     | 0.44              | 1.55             | ±      | 0.05                |
| D              | + D          | 46.15      | ±          | 0.92          | 10.56      | ±                     | 1.33              | 1.63             | ±      | 0.18                |
| Offline        | MLE          |            | 63.73      |               |            | 9.85                  |                   |                  | 1.90   |                     |
| Test           | set          |            | 70.94      |               |            | 0.00                  |                   |                  | 2.19   |                     |

#### <span id="page-15-1"></span>J. Sample preparation for human evaluations

The listening test consisted of pairwise comparisons between different systems. The pairwise comparisons are prepared as following: We randomly sample 32 melodies from the test set, and have each system generate an accompaniment to the first eight bars of a song. For each melody, the accompaniments from each system is paired with all other systems, yielding 4 2 = 6 pairs. Going through all 32 melodies, we obtain 6 ∗ 32 = 192 total pairs. We recruited ten musicians, who each rated 15 to 20 pairs.

## K. Quantitative Metrics Details

The entropy of chord length uses a base of e result in unit of nats. The EMD of chord-to-note onset interval reported in Table [1](#page-6-0) is multiplied by 10<sup>3</sup> for better comparison. To determine the distribution of chord-note onset intervals, we categorized these intervals into bins defined by the number of frames, using bin boundaries set at [0, 1, 2, ..., 16, 17, ∞]. Similarly, for the distribution of chord lengths, we organized the data into bins based on the number of frames, with bin boundaries established at [0, 1, 2, ..., 32, 33, ∞]. Following this, we generated histograms for both the chord-note onset intervals and chord lengths. Finally, we computed the Earth Mover's Distance (EMD) between these histograms.

For note-in-chord ratio of a song, we average across the binary value at each frame representing whether the note is in a chord. For calculating whether the note is in a chord, we exclude the frames where either input melody or output chord is silence. Then, to report the note-in-chord ratio of a system, we average across the note-in-chord ratio of each song. To report the note-in-chord ratio at certain beat reported in [§5.4](#page-7-1) and [§A,](#page-11-1) for each song we only consider the frames at that beat, and when averaging across songs, we exclude the song where the whole beat is silence.

## <span id="page-15-3"></span><span id="page-15-0"></span>L. Dataset and Data Representation Details

![](_page_15_Picture_7.jpeg)

Figure 7. A visualization of example data samples.

In ReaLchords, we represent each chord by a unique name, assigning each distinct chord name a unique index. Melody tokens x<sup>t</sup> indicate both pitch and whether they mark the onset of a new note or the continuation of the previous token. Similarly, chord tokens y<sup>t</sup> denote the chord symbol and its onset or continuation status. The Hooktheory dataset we use comprises 5041 distinct chords. For the output y, each frame y<sup>t</sup> can be one of three possibilities: a chord selected from the 5041 available options, a chord hold (indicating the continuation of the previous chord) from the same set of options, or a silence, which signifies the generation of silence in the current frame. Similarly, there are 128 possible melody pitch values, ranging from 0 to 127 according to the MIDI pitch standard. For the input x, each frame x<sup>t</sup> also falls into one of three categories: a note-on event chosen from 128 possible values, a note hold (indicating the continuation of the previous note) among the same 128 values, or a silence, indicating no input for the current frame.

As an example shown in Figure [7,](#page-15-3) the input melody x for that example at each frame would be: [C4 on, C4 hold, C4 hold, C4 hold, G4 on, G4 hold, G4 hold, G4 hold, A4 on, A4 hold, A4 hold, A4 hold, F4 on, F4 hold, F4 hold, F4 hold], and the output chords y for that example at each frame would be: [C on, C hold, C hold, C hold, G on, G hold, G hold, G hold, Am on, Am hold, Am hold, Am hold, F on, F hold, F hold, F hold].

The maximum length of the examples used for training is set to 256 frames, equivalent to 64 beats, with any samples exceeding this duration being randomly cropped during training.

## <span id="page-15-2"></span>M. Histogram Statistics of Compared Systems

In Figure [8](#page-16-0) and Figure [9,](#page-17-0) we present the histogram of chord-to-note onset interval, chord length and note-to-chord harmonic interval for one of each system compared in Table [1.](#page-6-0) Those histograms are used to calculate statistics report in Table [1.](#page-6-0)

<span id="page-16-0"></span>![](_page_16_Figure_1.jpeg)

Figure 8. The histogram of chord onset interval, chord length and harmonic interval for each system compared in [§5.3.](#page-6-1)

<span id="page-17-0"></span>![](_page_17_Figure_1.jpeg)

Figure 9. The histogram of chord onset interval, chord length and harmonic interval for each system compared in [§5.3.](#page-6-1)