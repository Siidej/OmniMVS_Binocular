
with open('./omnithings_train.txt', 'w') as f:
    f.truncate(0)
    for i in range(0, 7500):
        f.write(f'{i:05}.png\n')
f.close()
with open('./omnithings_val.txt', 'w') as f:
    f.truncate(0)
    for i in range(7500, 10000):
    #for i in range(10, 15):
        f.write(f'{i:05}.png\n')
f.close()