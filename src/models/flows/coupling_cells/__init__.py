"""Flows based on coupling cells

A coupling cell is a variable change x^i = f(y^j) such that there are two variable groups
M and N such that for all n in N and m in M
* x^n = y^n
* x^m = f(x^m,T(x^N)) <- each m transformed indepently and depends on m and all N
such that f(x,t^N) has a simple derivative wrt x.
"""