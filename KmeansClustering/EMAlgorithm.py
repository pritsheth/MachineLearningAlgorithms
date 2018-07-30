# -*- coding: utf-8 -*-
import math


def probFunc(x, m1, std1, m2, std2):
    p = []
    ax = []
    bx = []
    for i in x:
        px1 = (1 / (math.sqrt(2 * math.pi) * std1)) * math.exp(-(math.pow(i - m1, 2) / (2 * math.pow(std1, 2))))
        px2 = (1 / (math.sqrt(2 * math.pi) * std2)) * math.exp(-(math.pow(i - m2, 2) / (2 * math.pow(std2, 2))))
        print("Xi | cluster", px1, px2)

        probB = px1 / (px1 + px2)
        bx.append(probB)
        probA = 1 - probB
        ax.append(probA)
        print("Bi and Ai")
        print([probB, probA])
    return bx, ax


def computeU(b, x):
    deno = sum(b)
    num = [i * j for i, j in zip(b, x)]
    up = sum(num)
    print("new mean", up / deno)
    return up / deno


def computeV(u, b, x):
    deno = sum(b)
    num = [i * (j - u) ** 2 for i, j in zip(b, x)]
    up = sum(num)
    print("new variance", up / deno)


x = [-67, -48, 6, 8, 14, 16, 23, 24]
bx, ax = probFunc(x, -40, 20, 30, 40)
value = sum(bx) / 8
print("new value", value)
u1 = computeU(bx, x)
computeV(u1, bx, x)

u2 = computeU(ax, x)
computeV(u2, ax, x)

#
#
# Xi | cluster 0.00801916636709598 0.0005270946166416028
# Bi and Ai
# [0.9383245354144233, 0.06167546458557671]
#
#
# Xi | cluster 0.01841350701516617 0.0014898676517204018
# Bi and Ai
# [0.9251449728171419, 0.07485502718285808]
# Xi | cluster 0.0014163518870800589 0.008330615072294992
# Bi and Ai
# [0.14531206404857575, 0.8546879359514242]
# Xi | cluster 0.001119726514742145 0.0085735963754846
# Bi and Ai
# [0.11551523945117986, 0.8844847605488202]
# Xi | cluster 0.0005210467407211298 0.009206753507583085
# Bi and Ai
# [0.05356264802126879, 0.9464373519787312]
# Xi | cluster 0.00039577257914899825 0.00938100867292345
# Bi and Ai
# [0.040480866754086756, 0.9595191332459132]
# Xi | cluster 0.00013971292074397236 0.009822000236184483
# Bi and Ai
# [0.014024989330956679, 0.9859750106690434]
# Xi | cluster 0.00011920441007324213 0.009861983272697224
# Bi and Ai
# [0.011942908385443235, 0.9880570916145568]
# new mean -46.104154347017946
# new variance 722.2585585038782
# new mean 13.807537975942966
# new variance 167.0053793207507
#
# Bi and Ai
#
#
# Xi | cluster 0.00801916636709598 0.0005270946166416028
# [0.9383245354144233, 0.06167546458557671]
#
#
#
# Xi | cluster 0.01841350701516617 0.0014898676517204018
# Bi and Ai
# [0.9251449728171419, 0.07485502718285808]
#
#
#
# Xi | cluster 0.0014163518870800589 0.008330615072294992
# Bi and Ai
# [0.14531206404857575, 0.8546879359514242]
#
#
#
# Xi | cluster 0.001119726514742145 0.0085735963754846
# Bi and Ai
# [0.11551523945117986, 0.8844847605488202]
#
#
#
#
# Xi | cluster 0.0005210467407211298 0.009206753507583085
# Bi and Ai
# [0.05356264802126879, 0.9464373519787312]
#
#
#
#
# Xi | cluster 0.00039577257914899825 0.00938100867292345
# Bi and Ai
# [0.040480866754086756, 0.9595191332459132]
#
#
#
# Xi | cluster 0.00013971292074397236 0.009822000236184483
# Bi and Ai
# [0.014024989330956679, 0.9859750106690434]
#
#
#
# Xi | cluster 0.00011920441007324213 0.009861983272697224
# Bi and Ai
# [0.011942908385443235, 0.9880570916145568]
# new mean -46.104154347017946
# new variance 722.2585585038782
# new mean 13.807537975942966
# new variance 167.0053793207507
