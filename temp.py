import pbit
import turtle
import numpy as np
import math

def _dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def _not_drawn(i, j, drawn):
    for ii in drawn:
        if math.isclose(i, ii[0]) and math.isclose(j, ii[1]):
            return False
    return True


length = 50
Nm = 17

# setup
if(15 > Nm > 25):
    turtle.screensize(2000,2000)
turtle.speed("fastest")
turtle.penup()
turtle.setpos(0, 200)
turtle.pendown()
turn = 360 / Nm

J = np.array([[0, -0.4407, 0, 0, 0, 0],
              [-0.4407, 0, -0.4407, 0, 0, 0],
              [0, -0.4407, 0, 0, 0, 0],
              [0, 0, 0, 0, -0.4407, 0],
              [0, 0, 0, -0.4407, 0, -0.4407],
              [0, 0, 0, 0, -0.4407, 0]])
turtle.hideturtle()

# place pbits
pbitpos = np.zeros((Nm, 2))
for i in range(Nm):
    pbitpos[i, 0], pbitpos[i, 1] = turtle.pos()
    turtle.dot(size=40)
    turtle.penup()
    turtle.right(turn)
    turtle.forward(100)
print(pbitpos)

# draw weights
drawn = []
for i in range(Nm):  # source pbit
    for j in range(Nm):  # destination pbit
        if J[i, j]:
            turtle.penup()
            turtle.goto(pbitpos[i, 0], pbitpos[i, 1])
            turtle.pendown()
            turtle.goto(pbitpos[j, 0], pbitpos[j, 1])
            turtle.penup()

            # turtle.penup()
            # turtle.goto(pbitpos[i, 0], pbitpos[i, 1])
            # turtle.setheading(turtle.towards(pbitpos[j, 0], pbitpos[j, 1]))
            # turtle.forward(_dist(pbitpos[i], pbitpos[j]) / 2)
            # if (pbitpos[i, 1] > pbitpos[j, 1]):
            #     turtle.left(90)
            # else:
            #     turtle.right(90)
            # turtle.forward(10)
            # if _not_drawn(i, j, drawn):
            #     turtle.pendown()
            #     turtle.write(str(J[i, j]), font=("Calibri", 13, "normal"))
            #     turtle.penup()
            #
            # drawn.append([i, j])
# hold drawing until click
turtle.exitonclick()
