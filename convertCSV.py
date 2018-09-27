import pandas as pd

raw = pd.read_csv('./Data/port.csv', sep='\t', index_col='Date')

writer = pd.ExcelWriter('portAM.xlsx')
raw.to_excel(writer, 'Sheet1')
writer.save()
