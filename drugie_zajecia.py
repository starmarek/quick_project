import sys
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from tabulate import tabulate
except ModuleNotFoundError as e:
    print(e, "Prosze doinstalowac brakujce moduly:\npython -m pip install <modul>", sep="\n")
    sys.exit(1)

beta = 5

def init2(S, K1, K2):
    """
    funkcja tworzy sieć dwuwarstwową
    i wypełnia jej macierze wag wartościami losowymi
    z zakresu od -0.1 do 0.1

    parametry:  S – liczba wejść do sieci         / liczba wejść warstwy 1
                K1 – liczba neuronow w warstwie 1
                K2 – liczba neuronow w warstwie 2 / liczba wyjść sieci

    wynik:      W1 – macierz wag warstwy 1 sieci
                W2 – macierz wag warstwy 2 sieci
    """
    return np.random.rand(S + 1, K1) * 0.2 - 0.1, np.random.rand(K1 + 1, K2) * 0.2 - 0.1


def dzialaj2(W1, W2, X):
    """
    funkcja symuluje działanie sieci dwuwarstwowej

    parametry: W1 – macierz wag pierwszej warstwy sieci
               W2 – macierz wag drugiej warstwy sieci
               X –  wektor wejść do sieci
                    sygnał podany na wejście (sieci / warstwy 1)

    wynik:     Y1 – wektor wyjść warstwy 1 (przyda się podczas uczenia)
               Y2 – wektor wyjść warstwy 2 / sieci
                    sygnał na wyjściu sieci
    """
    global beta

    # warstwa 1
    X1 = np.vstack((-1, X))
    U1 = W1.T.dot(X1)
    Y1 = 1 / (1 + np.exp(-beta * U1))

    # warstwa 2
    X2 = np.vstack((-1, Y1))
    U2 = W2.T.dot(X2)
    Y2 = 1 / (1 + np.exp(-beta * U2))

    return Y1, Y2


def ucz2(W1przed, W2przed, P, T, n, m, e):
    """
    funkcja uczy sieć dwuwarstwową
    na podanym ciągu uczącym (P,T)
    przez zadaną liczbę kroków (n)

    parametry: W1przed – macierz wag warstwy 1 przed uczeniem
               W1przed – macierz wag warstwy 2 przed uczeniem
               P – ciąg uczący – przykłady - wejścia
               T – ciąg uczący – żądane wyjścia
                   dla poszczególnych przykładów
               n – liczba epok uczenia
               m – maksymalna liczba kroków uczenia
               e – błąd, który sieć ma osiągnąć

    wynik:     W1po – macierz wag warstwy 1 po uczeniu
               W2po – macierz wag warstwy 2 po uczeniu
    """
    liczbaPrzykladow = P.shape[1]

    W1 = W1przed
    W2 = W2przed

    S2 = W2.shape[0]

    wspMomentum = 0.7
    wspUcz = 0.1
    blad2poprzedni = 0
    dW1 = 0
    dW2 = 0
    global beta
    plot_data2 = {}
    plot_data1 = {}

    for i in range(1, n + 1):
        # losuj numer przykładu
        nrPrzykladu = np.random.randint(liczbaPrzykladow, size=1)

        # podaj przykład na wejścia i oblicz wyjścia
        X = P[:, nrPrzykladu]
        X1 = np.vstack((-1, X))
        Y1, Y2 = dzialaj2(W1, W2, X)

        X2 = np.vstack((-1, Y1))

        # oblicz błędy na wyjściach warstw
        D2 = T[:, nrPrzykladu] - Y2
        E2 = beta * D2  * Y2 * (1 - Y2)

        D1 = W2[1:S2, :] * E2
        E1 = beta * D1  * Y1 * (1 - Y1)

        # obsługa błędu średniokwadratowego i wczesnego końca nauczania
        blad2 = np.sum(D2 ** 2 / 2)
        blad1 = np.sum(D1 ** 2 / 2)
        plot_data2[i] = blad2
        plot_data1[i] = blad1
        if i >= m:
            # limit kroków został osiągnięty
            break
        elif blad2 <= e:
            # pożądany wymiar błędu został osiągnięty
            try:
                # Zakładamy że 40 poprzednich wartości błędu musi spełniać warunek:
                # wartość_błędu / pożądany_wymiar_błędu <= 10. Jeżeli którakolwiek wartość nie
                # spełni tego warunku uznajemy to za chwilową poprawę (min. lokalne) i kontynuujemy naukę sieci.
                if any(list(plot_data2.values())[i - b] / e >= 10 for b in range(2, 42)):
                    pass
                else:
                    break
            except IndexError:
                # Aby uniknąć błędów w początkowych stadiach nauki,
                # kiedy słownik z błędami jest jeszcze mały i istnieje
                # ryzyko index_out_of_range errora.
                pass

        # oblicz poprawki wag (momentum)
        dW1 = wspUcz * X1 * E1.T + wspMomentum * dW1
        dW2 = wspUcz * X2 * E2.T + wspMomentum * dW2

        # zastosuj poprawkę
        W1 = W1 + dW1
        W2 = W2 + dW2

        # adaptacyjny współczynnik uczenia
        if blad2 > 1.04 * blad2poprzedni and 0.7 * wspUcz >= 0.15:
            wspUcz = 0.7 * wspUcz
        else:
            wspUcz = 1.05 * wspUcz
        blad2poprzedni = blad2

    return W1, W2, plot_data1, plot_data2

# def testuj(P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia):
#     def single_run(P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia, g):

#         W1przed, W2przed = init2(2, 2, 1)
#         W1po, W2po, plot_data1, plot_data2 = ucz2(W1przed, W2przed, P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia, g)
#         Y1, Y2a = dzialaj2(W1po, W2po, P[:, [0]])
#         Y1, Y2b = dzialaj2(W1po, W2po, P[:, [1]])
#         Y1, Y2c = dzialaj2(W1po, W2po, P[:, [2]])
#         Y1, Y2d = dzialaj2(W1po, W2po, P[:, [3]])
#         return [Y2a, Y2b, Y2c, Y2d], plot_data1, plot_data2

#     arr = []
#     minval = {}
#     for i in [x / 100.0 for x in range(1, 50, 1)]:
#         g = i
#         print(i)
#         arr.clear()
#         for j in range(5):
#             Y, plt1, plt2 = single_run(P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia, g)
#             arr.append(max(plt1.keys()))
#         minval[g] = min(arr)
#     return minval

if __name__ == '__main__':
    # przygotowanie zmiennych
    P = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    T = np.array([[0, 1, 1, 0]])
    ilosc_petli_nauczania = 5000
    maks_ilosc_krokow_nauczania = 3500
    blad_do_osiagniecia = 0.0003
    # print(testuj(P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia))

    # stworzenie, uczenie i przetestowanie sieci
    W1przed, W2przed = init2(2, 2, 1)
    W1po, W2po, plot_data1, plot_data2 = ucz2(W1przed, W2przed, P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia)
    Y1, Y2a = dzialaj2(W1po, W2po, P[:, [0]])
    Y1, Y2b = dzialaj2(W1po, W2po, P[:, [1]])
    Y1, Y2c = dzialaj2(W1po, W2po, P[:, [2]])
    Y1, Y2d = dzialaj2(W1po, W2po, P[:, [3]])

    # zapisanie wyniku z testu w tablicy
    Y = [Y2a, Y2b, Y2c, Y2d]
    Ypo = np.array([[]])
    for i in Y:
        Ypo = np.append(Ypo, i, axis=1)

    # przygotowanie wykresu błędu średniokwadratowego
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(list(plot_data1.keys()), list(plot_data1.values()))
    ax[0].set_xlim(1, len(plot_data1.keys()))
    ax[0].grid()
    ax[0].set_title("MSE warstwa 1")
    ax[0].set_ylabel('Wartość błędu')
    ax[0].set_xlabel('Krok uczenia')

    ax[1].plot(list(plot_data2.keys()), list(plot_data2.values()))
    ax[1].set_xlim(1, len(plot_data2.keys()))
    ax[1].grid()
    ax[1].set_title("MSE warstwa 2")
    ax[1].set_ylabel('Wartość błędu')
    ax[1].set_xlabel('Krok uczenia')

    plt.tight_layout()

    # wyświetlenie wyniku oraz wykresu błędu
    print(tabulate(Ypo, tablefmt='fancy_grid'))
    plt.show()
