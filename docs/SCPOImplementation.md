## **1. Cost Computation (`cost` Tensor)**
The **cost function** is defined as a continuous function based on CGM target values, ensuring a **smooth penalty scaling** while prioritizing safety in hypoglycemic conditions. The cost is computed using:

\[
C(s_t) = \begin{cases}
0, & 70 \leq cgm_t \leq 180 \\  
\frac{70 - cgm_t}{8}, & 54 \leq cgm_t < 70  \\  
\frac{70 - cgm_t}{4}, & cgm_t < 54  \\  
\frac{cgm_t - 180}{140}, & 180 < cgm_t \leq 250  \\  
\frac{cgm_t - 180}{70}, & cgm_t > 250  
\end{cases}
\]

where:
- **Hypoglycemia** (low glucose) receives **stronger penalties** than **hyperglycemia** (high glucose).
- **Mild deviations** in hyperglycemia have **lower impact**, ensuring smooth optimization.
- **Normalization ensures stable training**, preventing extreme cost variations from destabilizing policy updates.
- This function aligns with **SCPO's goal of enforcing constraints on the maximum expected cost**.

To illustrate how costs are computed, the following table presents example CGM values covering all five ranges and their corresponding cost values before and after normalization:

| CGM Values (mg/dL)       | Category                | Raw Cost Values            | Normalized Cost Values     |
|--------------------------|------------------------|----------------------------|----------------------------|
| 45, 50, 52, 53          | Severe Hypoglycemia    | 6.25, 5.00, 4.50, 4.25     | 1.00, 0.80, 0.72, 0.68     |
| 55, 60, 65, 69          | Mild Hypoglycemia      | 1.88, 1.25, 0.63, 0.13     | 0.38, 0.25, 0.13, 0.03     |
| 80, 100, 140, 175       | Normoglycemia          | 0.00, 0.00, 0.00, 0.00     | 0.00, 0.00, 0.00, 0.00     |
| 190, 200, 220, 240      | Mild Hyperglycemia     | 0.07, 0.14, 0.29, 0.43     | 0.02, 0.03, 0.06, 0.09     |
| 260, 270, 280, 300      | Severe Hyperglycemia   | 1.14, 1.29, 1.43, 1.71     | 0.23, 0.26, 0.29, 0.35     |

## **2. State-wise Cost Increment (`D_i` Tensor)**
In SCPO, we track the **maximum encountered state-wise cost (`M`)** along a trajectory and compute the cost increment (`D_i`) as:

\[
D_i(s_t, a_t, s_{t+1}) = \max\{ C(s_t) - M_t, 0 \}
\]

where:
- **`M_t` represents the maximum cost encountered so far**.
- **`D_i` ensures that cost increments are non-negative**.
- This prevents unnecessary constraint violations in policy optimization.

## **3. Cumulative Cost Return (`J_Di` Tensor)**
The expected maximum cost return for a policy \(\pi\) is given by:

\[
J_{D_i}(\pi) = E_{\tau \sim \pi} \left[ \sum_{t=0}^{H} D_i(s_t, a_t, s_{t+1}) \right]
\]

which is stored in the `cost_return` tensor:
- If **`t = 0` or the episode starts**, then \( J_{D_i}(s_0) = D_i(s_0) \).
- Otherwise:
  \[
  J_{D_i}(s_t) = J_{D_i}(s_{t-1}) + D_i(s_t)
  \]
- This ensures **efficient tracking of cumulative state-wise costs**.

## **4. Cost Advantage (`A_{D_i}` Tensor)**
We compute the **Generalized Advantage Estimation (GAE)** for costs:

\[
A_{D_i}(s_t, a_t) = \sum_{t=0}^{H} \gamma^t (D_i(s_t, a_t, s_{t+1}) + \gamma J_{D_i}(s_{t+1}) - J_{D_i}(s_t))
\]

To **normalize** the cost advantage, we apply:

\[
A_{D_i} \leftarrow \frac{A_{D_i}}{\max(A_{D_i}) + 1e^{-5}}
\]

This ensures **stable updates** in policy optimization and prevents large cost fluctuations from dominating learning.

---

In the following sections, we will describe **how these tensors are implemented in the SCPO optimization process** and how they interact with the trust region constraints to ensure state-wise safety compliance.

