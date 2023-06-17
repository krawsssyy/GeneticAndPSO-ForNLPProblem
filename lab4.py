from time import perf_counter
import argparse
from random import random, uniform, choice, randint
from numpy.random import uniform as npuniform


def fitNLP(x):
    return sum([sum([x[j] for j in range(i + 1)])**2 for i in range(len(x))])


def randNLP(n):
    return [uniform(-65.536, 65.536) for _ in range(n)]


def getBestNLP(P):
    bestFit = 10000000000000000
    bestInd = -1
    for i in range(len(P)):
        if P[i][1] < bestFit:
            bestFit, bestInd = P[i][1], i
    return [bestFit, bestInd]


def getAvgNLP(P):
    return sum([P[i][1] for i in range(len(P))]) / len(P)


def propSelectionNLP(sums):
    g = random()
    return [i for i in range(len(sums)) if g < sums[i]][0]


def getParentsViaPropSelectNLP(P):
    parents = []
    total = sum([P[x][1] for x in range(len(P))])
    props = [(P[x][1] / total, x) for x in range(len(P))]
    sums = [props[i][0] + sum([props[j][0] for j in range(i)]) for i in range(len(props))]
    lst = set([i for i in range(len(P))])
    while len(lst) != 0:
        g = choice(list(lst))
        parents.append(g)
        lst.remove(g)
        p2 = propSelectionNLP(sums)
        parents.append(props[p2][1])
    return parents


def completeMeanCrossNLP(sol1, sol2):
    return [(sol1[i] + sol2[i]) / 2 for i in range(len(sol1))]


def crossParentsNLP(P, parents):
    return [completeMeanCrossNLP(P[parents[i]][0], P[parents[i + 1]][0]) for i in range(0, len(parents), 2)]


def nonuniformMutationNLP(sol, p, r, t, T):
    ind = randint(0, len(sol) - 1)
    if p == 1:
        sol[ind] += (65.536 - sol[ind]) * (1 - r ** (t / T))
    else:
        sol[ind] -= (65.536 + sol[ind]) * (1 - r ** (t / T))
    return [y for y in sol]


def mutateCrossesNLP(crosses, t, T):
    muts = []
    for cross in crosses:
        p = choice([-1, 1])
        r = npuniform(0, 1)
        muts.append(nonuniformMutationNLP(cross, p, r, t, T))
    return muts


def miuplambdaSurvNLP(P, N, muts, descs):
    newPop = [(muts[i], fitNLP(muts[i])) for i in range(len(muts))]
    newPop.extend([(descs[i], fitNLP(descs[i])) for i in range(len(descs))])
    newPop.extend([y for y in P])
    return [y for y in sorted(newPop, key=lambda x: x[1], reverse=False)[:N]]


def nlpAE(N, M, T, n):
    Paux = [[uniform(-65.536, 65.536) for _ in range(n)] for _ in range(N)]
    P = [(Paux[i], fitNLP(Paux[i])) for i in range(N)]
    bestInGen = []
    avgsInGen = []
    t = 1
    parents = []
    descendents = []
    mutants = []
    bestInGen.append(getBestNLP(P)[0])
    avgsInGen.append(getAvgNLP(P))
    while t <= M:
        parents = getParentsViaPropSelectNLP(P)
        descendents = crossParentsNLP(P, parents)
        mutants = mutateCrossesNLP(descendents, t, T)
        P = miuplambdaSurvNLP(P, N, mutants, descendents)
        t += 1
        bestInGen.append(getBestNLP(P)[0])
        avgsInGen.append(getAvgNLP(P))
    return P[getBestNLP(P)[1]][0], getBestNLP(P)[0], bestInGen, avgsInGen


def getAvgPSO(P):
    return sum([P[i][2] for i in range(len(P))]) / len(P)


def updateParticle(P, Vmax):
    for j in range(len(P[0])):
        if P[0][j] > 65.536 or P[0][j] < -65.536:
            P[0][j] = uniform(-65.536, 65.536)
    for j in range(len(P[1])):
        if P[1][j] > Vmax or P[1][j] < -Vmax:
            P[1] = [uniform(-2, 2) for _ in range(len(P[1]))]
            break
    newFit = fitNLP(P[0])
    if newFit != P[2]:
        P[2] = newFit
    if newFit < fitNLP(P[3]):
        P[3] = [y for y in P[0]]
    return P


def getGlobalBestPSO(P):
    gind = -1
    gfit = 999999999
    for i in range(len(P)):
        if P[i][2] < gfit:
            gfit, gind = P[i][2], i
    return [y for y in P[gind][0]], [y for y in P[gind][1]], gfit, [y for y in P[gind][3]]


def PSO(n, M, T, w, c1, c2, Vmax):
    pos = [[uniform(-65.536, 65.536) for _ in range(n)] for i in range(M)]
    speed = [[uniform(-2, 2) for _ in range(n)] for i in range(M)]
    P = [[pos[i], speed[i], fitNLP(pos[i]), pos[i]] for i in range(M)]
    gbest = getGlobalBestPSO(P)
    gbestList = []
    gbestList.append(gbest)
    avgsInIter = []
    avgsInIter.append(getAvgPSO(P))
    t = 1
    while t <= T:
        for i in range(len(P)):
            P[i][1] = [w*P[i][1][j] + c1*random()*(P[i][3][j] - P[i][0][j]) + c2*random()*(gbest[0][j] - P[i][0][j]) for j in range(n)]
            P[i][0] = [P[i][0][j] + P[i][1][j] for j in range(n)]
            P[i] = updateParticle(P[i], Vmax)
        gbest = getGlobalBestPSO(P)
        gbestList.append(gbest)
        t += 1
        avgsInIter.append(getAvgPSO(P))
    return min(gbestList, key=lambda x: x[2]), [y[2] for y in gbestList], avgsInIter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='subcommand')
    sub.required = True
    parser_NLP = sub.add_parser('nlp')
    parser_NLP.add_argument('--n', type=int, required=True, help='Valoarea lui n(numarul de dimensiuni ale spatiului de cautare)')
    parser_NLP.add_argument('--N', type=int, required=True, help='Numarul de indivizi din populatie.')
    parser_NLP.add_argument('--M', type=int, required=True, help='Numarul maxim de generatii al algoritmului')
    parser_NLP.add_argument('--T', type=int, required=True, help='Indicele generatiei la care se doreste oprirea mutatiei')
    parser_PSO = sub.add_parser('pso')
    parser_PSO.add_argument('--n', type=int, required=True, help='Valoarea lui n(numarul de dimensiuni ale spatiului de cautare)')
    parser_PSO.add_argument('--M', type=int, required=True, help='Numarul de particule din populatie.')
    parser_PSO.add_argument('--T', type=int, required=True, help='Numarul maxim de iteratii al algoritmului')
    parser_PSO.add_argument('--w', type=float, required=True, help='Factorul de inertie')
    parser_PSO.add_argument('--c1', type=float, required=True, help='Factorul de invatare cognitiva')
    parser_PSO.add_argument('--c2', type=float, required=True, help='Factorul de invatare sociala')
    parser_PSO.add_argument('--Vm', type=float, required=True, help='Viteza maxima a unei particule(in orice directie)')

    args = parser.parse_args()
    if args.subcommand == "nlp":
        if 1 > args.N or 1 > args.M or 1 > args.T:
            print("Argumentele N, M si T trebuie sa fie pozitive si mai mari decat 1!")
            exit(1)
        if 2 > args.n:
            print("Dimensiunea nu poate fi mai mica de 2!")
            exit(1)
        best = []
        bestFit = 99999999999
        solsFit = []
        worst = []
        worstFit = -1
        totalTime = 0
        bestsFinal, avgsFinal = [], []
        with open('output_nlp-' + str(args.n) + '-' + str(args.N) + '-' + str(args.M) + '-' + str(args.T) + '.txt', 'w') as f:
            for i in range(10):
                time0 = perf_counter()
                sol, solFit, bests, avgs = nlpAE(args.N, args.M, args.T, args.n)
                time1 = perf_counter()
                totalTime = totalTime + (time1 - time0)
                print("i = %d done" % i)
                print("total time atm : " + str(totalTime))
                f.write("Solutie " + str(i + 1) +" : " + str(sol) + "\n")
                f.write("Fitness solutie " + str(i + 1) + " : " + str(solFit) + "\n")
                f.write("Solutie generata in " + str(time1 - time0) + " secunde\n")
                if solFit < bestFit:
                    best = [x for x in sol]
                    bestFit = solFit
                    bestsFinal = [x for x in bests]
                    avgsFinal = [y for y in avgs]
                if solFit > worstFit:
                    worst = [x for x in sol]
                    worstFit = solFit
                solsFit.append(solFit)
            avg = 0
            for x in solsFit:
                avg = avg + x
            avg = avg / 10
            f.write("\n")
            f.write("Cea mai buna solutie : " + str(best) + "\n Fitness : " + str(bestFit) + "\n")
            f.write("Besturile acestei solutii : " + str(bestsFinal) + "\n" + "Averageruile acestei solutii : " + str(avgsFinal) + "\n")
            f.write("Cea mai proasta solutie : " + str(worst) + "\n Fitness : " + str(worstFit) + "\n")
            f.write("Valoarea medie a solutiilor : " + str(avg) + "\n")
            f.write("Timp petrecut pentru generarea solutiilor : " + str(totalTime) + " secunde")
        exit(0)
    elif args.subcommand == "pso":
        if 1 > args.M or 1 > args.T:
            print("Numarul de particule si numarul de iteratii nu pot fi mai mici decat 1!")
            exit(1)
        if 2 > args.n:
            print("Dimensiunea nu poate fi mai mica de 2!")
            exit(1)
        if 0 >= args.w:
            print("Factorul de inertie nu poate fi negativ!")
            exit(1)
        if 0 >= args.c1 or 0 >= args.c2:
            print("Factorii de invatare trebuie sa fie mai mari decat 0!")
            exit(1)
        if 2 > args.Vm:
            print("Viteza maxima nu poate fi mai mica decat 2!")
            exit(1)
        best = []
        bestFit = 999999
        solsFit = []
        worst = []
        worstFit = 0
        totalTime = 0
        bestsFinal = []
        avgsFinal = []
        with open('output_pso-' + str(args.n) + '-' + str(args.M) + '-' + str(args.T) + '-' + str(args.w) + '-' + str(args.c1) + '-' + str(args.c2) + '-' + str(args.Vm) + '.txt', 'w') as f:
            for i in range(10):
                time0 = perf_counter()
                sol, bests, avgs = PSO(args.n, args.M, args.T, args.w, args.c1, args.c2, args.Vm)
                time1 = perf_counter()
                print("i = %d done" % i)
                totalTime = totalTime + (time1 - time0)
                print("total time atm : " + str(totalTime))
                f.write("Solutie " + str(i + 1) + " : " + str([y for y in sol[0]]) + "\n")
                f.write("Fitness solutie " + str(i + 1) + " : " + str(sol[2]) + "\n")
                f.write("Solutie generata in " + str(time1 - time0) + " secunde\n")
                if sol[2] < bestFit:
                    best = [x for x in sol[0]]
                    bestFit = sol[2]
                    bestsFinal = [y for y in bests]
                    avgsFinal = [y for y in avgs]
                if sol[2] > worstFit:
                    worst = [x for x in sol[0]]
                    worstFit = sol[2]
                solsFit.append(sol[2])
            avg = 0
            for x in solsFit:
                avg = avg + x
            avg = avg / 10
            f.write("\n")
            f.write("Cea mai buna solutie : " + str([y + 1 for y in best]) + "\n Fitness : " + str(bestFit) + "\n")
            f.write("Besturile acestei solutii : " + str(bestsFinal) + "\n" + "Averageruile acestei solutii : " + str(avgsFinal) + "\n")
            f.write("Cea mai proasta solutie : " + str([y + 1 for y in worst]) + "\n Fitness : " + str(worstFit) + "\n")
            f.write("Valoarea medie a solutiilor : " + str(avg) + "\n")
            f.write("Timp petrecut pentru generarea solutiilor : " + str(totalTime) + " secunde")
        exit(0)
    else:
        print("Valoare gresita pentru algoritm!Iesire...")
        exit(1)
