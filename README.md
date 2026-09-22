
$$\mathcal{L}{\text{DAN-DG}} = \mathcal{L}{\text{ERM}} + \frac{\lambda_{\text{DG}}}{3}\sum_{e<e'}\text{MMD}^2\big(F(X_e), F(X_{e'})\big)$$

$$\text{MMD}^2 = \underbrace{\overline{k(s,s')}}{\text{source–source}} + \underbrace{\overline{k(t,t')}}{\text{target–target}} - 2\underbrace{\overline{k(s,t)}}_{\text{source–target}}$$
