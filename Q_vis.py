maze=[
    [0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0],
    [0, 9, 9, 9, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0],
]

maze_Q = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]


mx = 0
my = 0
goal = (4, 4)
reward = 5
penalty = -1
alpa = 0.5

def move(mx, my):
    action=input("w,a,s,d")
    if action == "w":
        mx -= 1
        return mx, my
    elif action == "s":
        mx += 1
        return mx, my
    elif action == "a":
        my -= 1
        return mx, my
    elif action == "d":
        my += 1
        return mx, my
    else:
        print("Invalid action. Please use w, a, s, or d.")
        return move(mx, my)
