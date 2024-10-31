import pandas as pd


class Borehole:
    def __init__(self, name, link=f'Z:\\2-UGIR\ONtrim\Нестеров С\Конференция КНТК 2024\Filter.xlsx'):
        self.name = name
        self.link = link

        self.df = pd.read_excel(self.link, sheet_name=self.name)
        self.df.set_index(self.df.columns[0], inplace=True)
        self.df = self.df.iloc[:, 1:]
        self.params = list(self.df.columns)

        self.df.dropna(inplace=True)
        self.df['Cluster'] = 0

        self.df_list = [self.df.copy()]
