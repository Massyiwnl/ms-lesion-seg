"""Regolarizzazione dello Jacobiano (robustezza / vincolo di Lipschitz).

Penalizzando la norma di Frobenius dello Jacobiano dOutput/dInput si limita di quanto
l'uscita può cambiare a fronte di piccole perturbazioni dell'ingresso: il modello
diventa localmente più "piatto" e quindi meno sensibile a variazioni di contrasto,
rumore e disomogeneità di campo, cioè esattamente ciò che cambia tra scanner diversi.

||J||_F^2 costerebbe un backward per ogni canale d'uscita. Si usa lo stimatore di
Hutchinson (Hoffman et al., 2019): con v casuale sulla sfera unitaria,
E[||J^T v||^2] = ||J||_F^2 / C, quindi bastano 1-2 proiezioni. La costante C viene
assorbita nel coefficiente `regularization.jacobian_lambda`.
"""
from __future__ import annotations
import torch


def jacobian_regularization(logits, inputs, n_proj: int = 1):
    """Stima di ||dOutput/dInput||_F^2 con proiezioni casuali.

    `inputs` deve avere requires_grad=True e `logits` derivare da esso.
    Se il modello usa deep supervision, si regolarizza solo l'uscita principale.
    """
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    reg = logits.new_zeros(())
    for _ in range(max(1, int(n_proj))):
        v = torch.randn_like(logits)
        v = v / (v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1))) + 1e-12)
        (grad,) = torch.autograd.grad(outputs=logits, inputs=inputs, grad_outputs=v,
                                      create_graph=True, retain_graph=True)
        reg = reg + grad.flatten(1).pow(2).sum(dim=1).mean()
    return reg / max(1, int(n_proj))
