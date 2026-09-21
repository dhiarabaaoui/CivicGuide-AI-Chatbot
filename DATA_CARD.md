# Data Card

## Dataset

The research pipeline uses **MultiDoc2Dial**, a document-grounded dialogue
dataset introduced by Feng et al. at EMNLP 2021. The project works with four
public-service domains:

- Department of Motor Vehicles (DMV);
- Social Security Administration (SSA);
- Veterans Affairs (VA);
- federal student aid (StudentAid).

The audited local corpus contains 488 document entries representing 475 unique
contents, 35,659 annotated source spans and 33,869 dialogue examples.

## Project transformations

The local pipeline:

1. validates document, span and dialogue references;
2. normalizes records into a canonical schema;
3. creates structure-aware document chunks;
4. maps gold spans to retrievable chunks;
5. computes lexical and dense retrieval artifacts;
6. freezes only the chunks, documents, vectors and index needed at runtime.

Raw downloads, processed datasets, embedding caches and generated evaluation
reports are excluded from Git. `runtime_artifacts/` contains the derived subset
needed by the deployed search service and a manifest with file hashes.

## Splits and leakage controls

The retrieval benchmark contains 24,603 training examples, 4,699 validation
examples and 4,567 test examples. Conversation identifiers are kept disjoint
between splits. Repeated text and document aliases were audited to reduce
leakage, and the test split was not used for routine prompt iteration.

## Additional validation material

The project also prepared 80 external validation tasks from official public
documents, balanced across the four domains. Their source material, human
reference delivery and automated candidate scoring are described in
`GUIDE_MEMOIRE_PROJET.md`. Generated validation files are not part of the Git
repository.

## Privacy and sensitive information

MultiDoc2Dial contains public documents and task-oriented dialogue examples.
The public application is not designed to collect or store personal case data.
Browser chat history is stored locally on the visitor's device. Users should
not submit Social Security numbers, claim identifiers, medical records or other
sensitive personal information.

## License and attribution

The official IBM MultiDoc2Dial repository is distributed under Apache-2.0 and
the Hugging Face dataset metadata also identifies Apache-2.0. Government source
documents and third-party dependencies retain their respective terms. This
project's MIT license applies only to original project code and documentation.

Official resources:

- <https://github.com/IBM/multidoc2dial>
- <https://huggingface.co/datasets/IBM/multidoc2dial>
- <https://doc2dial.github.io/multidoc2dial/>

## Citation

```bibtex
@inproceedings{feng2021multidoc2dial,
  title     = {MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents},
  author    = {Feng, Song and Patel, Siva Sankalp and Wan, Hui and Joshi, Sachindra},
  booktitle = {Proceedings of EMNLP},
  year      = {2021}
}
```

