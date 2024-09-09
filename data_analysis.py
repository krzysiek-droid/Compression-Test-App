import pandas as pd


def read_compression_test_data(file_path, start_row):
    """
    Parametry:
    file_path (str): Ścieżka do pliku CSV.
    start_row (int): Numer wiersza, od którego mają być odczytane dane (indeksowanie od 0).
    Zwraca:
    pd.DataFrame: DataFrame zawierający odczytane dane.
    """
    # Odczyt danych z pliku CSV z pominięciem wierszy do start_row
    data = pd.read_csv(file_path, sep=';', skiprows=start_row)

    return data

