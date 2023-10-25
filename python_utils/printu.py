def print_big(s, *args, **kwargs):
  pad = '=' * 30
  print(f'{pad} {s} {pad}')

def print_head(s, width=60):
  N_space = (width - len(s)) // 2
  print('=' * width)
  print(' ' * N_space + s + ' ' * N_space)
  print('=' * width)
