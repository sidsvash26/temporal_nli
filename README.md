# Temporal Reasoning in NLI
This repo contains the scripts to build the Temporal NLI dataset and also to run different models on it as described in the following paper:
> Vashishtha, SIddharth, Adam Poliak, Yash Kumar Lal, Benjamin Van Durme, Aaron Steven White. [Temporal Reasoning in Natural Language Inference](https://www.aclweb.org/anthology/2020.findings-emnlp.363/). Findings of the Association for Computational Linguistics: EMNLP 2020, November, 2020. 

```latex
@inproceedings{vashishtha-etal-2020-temporal,
    title = "Temporal Reasoning in Natural Language Inference",
    author = "Vashishtha, Siddharth  and
      Poliak, Adam  and
      Lal, Yash Kumar  and
      Van Durme, Benjamin  and
      White, Aaron Steven",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.363",
    pages = "4070--4078",
    abstract = "We introduce five new natural language inference (NLI) datasets focused on temporal reasoning. We recast four existing datasets annotated for event duration{---}how long an event lasts{---}and event ordering{---}how events are temporally arranged{---}into more than one million NLI examples. We use these datasets to investigate how well neural models trained on a popular NLI corpus capture these forms of temporal reasoning.",
}
```

# Python Environment
We use `pipenv` to run our scripts in a Python virtualenv. You can replicate the environment by cloning this repo and running the following from the root dir of this repo:

```bash
pipenv install --ignore-pipfile
```

If you don't have pipenv, you can install it by running:
```bash
pip install pipenv
```

# Dataset Creation
There are two steps to creating our recasted datasets:
1. Download the original datasets. Instructions [here](https://github.com/sidsvash26/temporal_nli/tree/main/data)
2. Run recasting scripts. Instructions [here](https://github.com/sidsvash26/temporal_nli/tree/main/src/recasting)

# Train from Scratch or Evaluate best models
To train on our models from scratch or to use our best models, follow instructions [here] (https://github.com/sidsvash26/temporal_nli/tree/main/src/recasting). Our roberta models can be downloaded [here](https://www.dropbox.com/s/f5tt2vpbfklr91u/roberta-large-temporal-nli.tar.gz?dl=0)


 
