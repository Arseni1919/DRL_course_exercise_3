print('Hello')
import random
# print(random.randint(0,1))
def foo(*args):
    print(*args)

foo(1,2,'3', True)

l = [1,2,3]
l.append(4)
print(l)
l.pop(0)
print(l)
# import itertools
# import time
# for t in itertools.count():
#     print(t)
#     time.sleep(1)

# # SAVE
# torch.save(model.state_dict(), PATH)
# # LOAD
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()