import pandas as pd

HEADER = ['Country', 'Population', 'Yearly_Change', 'Net_Change', 'Density (P / Km2)', 'Land_Area', 'Migrants',
          'Fertality_Rate', 'Median_Age', 'Urban_Population', 'World_Share']

Data: pd.DataFrame = pd.read_csv(filepath_or_buffer='F:\\2018_Spring\Programming\Python\Tensorflow\Data\country-pop.csv',
                                 delimiter='\t', index_col=None, header=None, names=HEADER, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

print(Data.head(10))