maze = [
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

mx, my = 0, 0
reward = 5
penalty = -1
alpa = 0.5

def move():
    global mx, my
    action = input("w,a,s,d: ")
    if action == "w" and mx > 0 and maze[mx - 1][my] != 9:
        mx -= 1
    elif action == "s" and mx < len(maze) - 1 and maze[mx + 1][my] != 9:
        mx += 1
    elif action == "a" and my > 0 and maze[mx][my - 1] != 9:
        my -= 1
    elif action == "d" and my < len(maze[0]) - 1 and maze[mx][my + 1] != 9:
        my += 1
    else:
        print("Invalid move or boundary reached. Try again.")
        return move()
    return [mx, my]

def print_grid(grid):
    for row in grid:
        print("    ".join(str(cell) for cell in row))

while True:
    x, y = mx, my
    position = move() 
      
    if maze[mx][my] == 0:
        maze_Q[x][y] = (1 - alpa) * maze_Q[x][y] + alpa * (maze_Q[mx][my] + penalty)
    
    if mx == 4 and my == 4:
        maze_Q[x][y] = (1 - alpa) * maze_Q[x][y] + alpa * (maze_Q[mx][my] + reward)
        print("Goal reached!")
        mx = 0
        my = 0
        print("Resetting position.")

    print_grid(maze_Q)
    print("現在の位置は"+str(mx)+","+str(my)+"です")
