import pandas as pd
import numpy as np
import numexpr as ne
import time

df = pd.DataFrame({
    'A' : np.random.rand(10000000),
    'B' : np.random.rand(10000000),
    'C' : np.random.rand(10000000),
    'D' : np.random.rand(10000000)
})

start_pd = time.time()
df['E'] = df['A'] + df['B'] + np.sin(df['C']) - np.log(df['D'])
df['F'] = np.where(df['A'] > 0.5, df['B'] * df['C'], df['D'] / df['A'])
df['G'] = (df['E'] ** 2 +df['F'] **2) ** 0.5
end_pd = time.time()

start_ne = time.time()
A,B,C,D =df['A'].values, df['B'].values, df['C'].values, df['D'].values
E = ne.evaluate('A * B + sin(C) - log(D)')
F = ne.evaluate('where(A >0.5, B * C, D/A)')
G = ne.evaluate('(E ** 2 + F ** 2) ** 0.5')
df['E_ne'], df['F_ne'], df['G_ne'] = E,F,G
end_ne = time.time()

print('Pandas time: ', end_pd-start_pd)
print('NumExpr with Pandas time: ', end_ne-start_ne)

print(df[['E', 'E_ne', 'F', 'F_ne', 'G', 'G_ne']])
