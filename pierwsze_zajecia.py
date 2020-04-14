#!/usr/bin/env python3
import sys
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from tabulate import tabulate
except ModuleNotFoundError as e:
    print(e, "Prosze doinstalowac brakujace moduly:\npython -m pip install <modul>", sep="\n")
    sys.exit(1)

beta = 5

def init1(S, K):
    """
    funkcja tworzy sieć jednowarstwową
    i wypełnia jej macierz wag wartościami losowymi
    z zakresu od -0.1 do 0.1

    parametry: S – liczba wejsć do sieci
               K – liczba neuronów w warstwie

    wynik:     W – macierz wag sieci
    """
    return np.random.rand(S, K) * 0.2 - 0.1


def dzialaj1(W, X):
    """
    funkcja symuluje działanie sieci jednowarstwowej

    parametry: W – macierz wag sieci
               X – wektor wejść do sieci
                   sygnał podany na wejście

    wynik:     Y – wektor wyjść sieci
               sygnał na wyjściu sieci
    """
    global beta
    U = W.T.dot(X)
    return 1 / (1 + np.exp(-beta * U))


def ucz1(Wprzed, P, T, n, m, e):
    """
    funkcja uczy sieć jednowarstwową
    na podanym ciągu uczącym (P,T)
    przez zadaną liczbę kroków (n)

    parametry: Wprzed – macierz wag sieci przed uczeniem
               P – ciąg uczący – przykłady - wejścia
               T – ciąg uczący – żądane wyjścia
                   dla poszczególnych przykładów
               n – liczba kroków
               m – maksymalna liczba kroków uczenia
               e – błąd, który sieć ma osiągnąć

    wynik:     Wpo – macierz wag sieci po uczeniu
    """
    liczbaPrzykladow = P.shape[1]
    W = Wprzed
    wspUcz = 0.1
    global beta
    plot_data = {}

    for i in range(1,  n + 1):
        # losuj numer przykładu
        nrPrzykladu = np.random.randint(liczbaPrzykladow, size=1)

        # podaj przykład na wejścia i oblicz wyjścia
        X = P[:, nrPrzykladu]
        Y = dzialaj1(W, X)

        # oblicz blędy na wyjściach
        D = T[:, nrPrzykladu] - Y
        E = D * beta * Y * (1 - Y)

        # obsługa błędu średniokwadratowego i wczesnego końca nauczania
        blad = np.sum(D ** 2 / 2)
        plot_data[i] = blad
        if i >= m:
            # limit kroków został osiągnięty
            break
        elif blad <= e and i >= 10:
            # Pożądany wymiar błędu został osiągnięty.
            # Dodany został dodatkowy warunkek, że liczba
            # kroków nie może być mniejsza niż 10. Poniżej tego poziomu
            # sięć pomimo zadowalającego wymiaru błędu zdaje się być niestabilna
            # i błąd po kolejnym kroku okazałby się pewnie ponownie zbyt duży.
            break

        # oblicz poprawki wag
        dW = wspUcz * X * E.T

        # zastosuj poprawkę
        W = W + dW

    return W, plot_data


if __name__ == '__main__':
    # przygotowanie zmiennych
    P = np.array([[4, 2, -1],
                  [0.01, -1, 3.5],
                  [0.01, 2, 0.01],
                  [-1, 2.5, -2],
                  [-1.5, 2, 1.5]])
    T = np.eye(3)
    ilosc_petli_nauczania = 70
    maks_ilosc_krokow_nauczania = 40 # TA WARTOŚĆ NIE MOŻE BYĆ MNIEJSZA NIŻ 10. JEST TO TOTALNE MINIMUM ŻEBY SIEĆ DAŁA RZETELNY WYNIK
    blad_do_osiagniecia = 0.0002

    # stworzenie, uczenie i przetestowanie sieci
    Wprzed = init1(5, 3)
    Wpo, plot_data = ucz1(Wprzed, P, T, ilosc_petli_nauczania, maks_ilosc_krokow_nauczania, blad_do_osiagniecia)
    Ypo = dzialaj1(Wpo, P)

    # przygotowanie wykresu błędu średniokwadratowego
    fig, ax = plt.subplots(1,1)
    ax.plot(list(plot_data.keys()), list(plot_data.values()))
    ax.set_xlim(1, len(plot_data.keys()))
    ax.grid()
    ax.set_title("Sumaryczny błąd średniokwadratowy")
    ax.set_ylabel('Wartość błędu')
    ax.set_xlabel('Krok uczenia')

    # wyświetlenie wyniku oraz wykresu błędu
    print(tabulate(Ypo, tablefmt='fancy_grid', headers=['Przykład 1', 'Przykład 2', 'Przykład 3'], showindex=['Ssak', 'Ptak', 'Ryba']))
    plt.show()

""" SPRAWOZDANIE Z SIECI JEDNOWARSTWOWYCH

Stworzona sieć bardzo ładnie osiąga zamierzony efekt, co widać po generowanej tabelce wynikowej.

Poza ograniczeniami z instrukcji (m - maksymalna liczba kroków uczenia | e - błąd, który sieć ma osiągnąć),
dodany został również warunek na minimalną ilość kroków -> ilość kroków nie może być mniejsza niż 10. Ograniczenie 
wynika z obserwacji, tj. nawet jeżeli sieć osiągała pożądany błąd przed 10 krokiem było to nierzetelne -> wyniki odbiegały od zamierzonych
a na wykresie było widać, że sieć dopiero co się ustabilizowała. Biorąc pod uwagę zachowanie uczącej się sieci -> prawdopodobnie z kolejnym krokiem błąd by wzrósł
a sieć sama w sobie jest na razie 'niestabilna'.

Sieć stabilizuje się średnio przy ok. 15 kroku. Wiadomo, że niekiedy ta wartość jest znacznie większa (osiąga górne ograniczenie) ale ostatecznie wyniki są zawsze
zadowalające.
"""