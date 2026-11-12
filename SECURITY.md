# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability
reporting: on https://github.com/Inferensys/bulwark, open the **Security** tab and choose
**Report a vulnerability**. Do not open a public issue, pull request or discussion for a
security problem.

Include what you found, the affected files or versions, steps to reproduce, and the
impact you expect. We will acknowledge the report, keep you informed while we
investigate, and credit you in the fix unless you prefer not to be named.

## Scope

In scope:

- the Python package (`bulwark/`), its command-line interface and the scripts in
  `scripts/`;
- the job and cloud scripts in `infra/` (for example anything that could leak
  credentials, storage access tokens or data, or leave cloud resources running);
- released model files and checkpoints.

Out of scope: vulnerabilities in third-party dependencies or upstream models, which
should go to their maintainers (tell us too if Bulwark is affected), and questions about
model accuracy or misuse, which belong in a normal issue.

## Supported versions

Bulwark is early research code. Fixes go to the `main` branch and the next release;
older versions are not patched.

## Safe use of model files

- PyTorch checkpoints (`.pt`) are Python pickles. Bulwark reads them with
  `torch.load(weights_only=True)`, which restricts unpickling to tensors and plain Python
  types, and requires PyTorch 2.6 or later, which fixed a known bypass of that restriction
  (CVE-2025-32434). That narrows the risk without removing it: load checkpoints only from
  sources you trust. The official model repositories hold no `.pt` files:
  https://huggingface.co/Inferensys/bulwark-b1-driver-monitoring holds a B1 release
  directory, whose weights are in safetensors (which stores no code), and
  https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx holds E1's ONNX
  graphs. Treat any `.pt` file said to come from an official repository as untrusted.
- Verify the checksums published with a release (`SHA256SUMS` in the repository;
  `shasum -a 256 -c SHA256SUMS`) before using its weights or ONNX files.
- Training data and model inputs are video of people. Treat them as personal data:
  restrict access, and do not attach them to issues or reports.
