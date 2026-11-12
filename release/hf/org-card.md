# Inferensys

Inference Systems (Inferensys) builds production AI systems for enterprise teams.

## Driver monitoring models

Bulwark is a family of research models that detect signs of driver drowsiness and distraction in video from a camera facing the driver.

- [Bulwark-B1]({{ b1_url }}): the base video model, built on Meta's V-JEPA 2.1 encoder, for PyTorch.
{{#released e1}}
- [Bulwark-E1]({{ e1_url }}): a smaller version distilled from Bulwark-B1, in the ONNX format, that runs on CPU. Its first version, E1 v0, scores well below Bulwark-B1: use it to try CPU deployment, and Bulwark-B1 for research.

Both are research previews with indicative results, and both are in the collection {{ collection_link }}.
{{/released}}
{{#unreleased e1}}
- Bulwark-E1: a smaller version for CPU, in the ONNX (Open Neural Network Exchange) format, in development.

Bulwark-B1 is a research preview: its results are indicative, from 3 test people per dataset. It is in the collection {{ collection_link }}.
{{/unreleased}}

## Licensing

The research weights here are licensed under CC BY-NC 4.0, and their use is also subject to the training datasets' terms. They can't be used commercially, including pilots, proofs of concept (POCs) and customer evaluations. Inferensys builds commercially licensed models for your cameras and use case, trained on data cleared for commercial use.

[Contact Inferensys]({{ contact_url }}) or write to {{ contact_email }}.

## Code

The code is open source on GitHub: [Inferensys/bulwark]({{ github }}), under Apache-2.0, with the Meta encoder code it vendors under MIT. These licences cover the code only, not the weights.

## Contact

For commercially licensed models, questions or removal requests, [contact Inferensys]({{ contact_url }}) or write to {{ contact_email }}.
