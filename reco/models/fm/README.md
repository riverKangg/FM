## Factorization Machine
| [Overview](#Overview) | [Configuration](#Configuration) | [Implementation](#Implementation) | [Discussion](#Discussion) |
| :--: | :--: | :--: | :--: |

### Overview

**model structure**

$$
\hat{y}(x) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} h(v_i, v_j) x_i x_j
$$