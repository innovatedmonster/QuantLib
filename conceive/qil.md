# QIL的实现思路

## 公式一览

$\begin{array}{rl}Q_W:&w\xrightarrow{T_W}\hat{w}\xrightarrow{D}\bar{w}\\Q_X:&x\xrightarrow{T_X}\hat{x}\xrightarrow{D}\bar{x}.\end{array}\quad\quad(1)$

</br>$\bar{v}=\frac{\left\lceil\hat{v}\cdot q_D\right\rfloor}{q_D}\quad\quad\quad\quad\quad\quad(2)$

</br>$\left.\hat{w}=\left\{\begin{array}{cc}0&|w|<c_W-d_W\\\operatorname{sign}(w)&|w|>c_W+d_W\\(\alpha_W|w|+\beta_W)^\gamma\cdot\operatorname{sign}(w)&otherwise,\\\end{array}\right.\right.\text{(3)}$

</br>$\begin{aligned}
&th_{\Delta}^{p} =c_{\Delta}+d_{\Delta}+0.5d_{\Delta}/q_{\Delta}  \\
&&\text{(4)} \\
&th_{\Delta}^{c} =c_\Delta-d_\Delta-0.5d_\Delta/q_\Delta. 
\end{aligned}$

</br>$\left.\hat{x}=\left\{\begin{array}{cc}0&x<c_X-d_X\\1&x>c_X+d_X&\text{(5)}\\\alpha_Xx+\beta_X&otherwise,\end{array}\right.\right.$



## 伪代码
Algorithm 1 Training low bit-width network using parameterized quantizers
 $\overline{\textbf{Input:Trainingdata}}$
 Output: A low bit-width model with quantized weights

$\{\bar{w}_{l}\}_{l=1}^{L}$ and activation quantizers $\{c_{X_l},d_{X_l}\}_{l=1}^{L}$ l: procedure TRAINING
 2: Initialize the parameter set $\{P_l\}_{l=1}^L$ where $P_l=$
 $\{w_l,c_{W_l},d_{W_l},\gamma_l,c_{X_l},d_{X_l}\}$
 3: for $l=1,...,L\mathbf{do}$
 4: Compute $\bar{w}_l$ from $w_l$ using Eq. 3 and Eq. 2
 5: Compute $\bar{x}_l$ from $x_l$ using Eq. 5 and Eq. 2
Compute $\bar{w}_l*\bar{x}_l$ 6:
Compute the loss $\ell$ Compute the gradient w.r.t. the output $\partial\ell/\partial x_{L+1}$
 7:
for $l=L,...,1\mathbf{do}$ 8: 9: Given $\partial\ell/\partial x_{l+1}$,
Compute the gradient of the parameters in $P_l$ 10:
Update the parameters in $P_l$ 11:
Compute $\partial\ell/\partial x_l$ 12: 13: procedure DEPLOYMENT

for $l=1,...,L\mathbf{do}$ 14:
Compute $\bar{w_l}$ from $w_l$ using Eq. 3 and Eq. 2 15: 16: Deploy the low bit-width model $\{w_l,c_{X_l},d_{X_l}\}_{l=1}^L$

## 实现细节