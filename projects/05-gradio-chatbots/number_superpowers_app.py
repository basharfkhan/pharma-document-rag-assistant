"""Number superpowers: square, cube and factorial of a number, as a small Gradio app."""
import math

import gradio as gr

MAX_FACTORIAL = 1000  # 1000! already has 2,568 digits


def superpowers(number):
    if number is None:
        raise gr.Error("Please enter a number.")

    square = number ** 2
    cube = number ** 3

    # Factorial is only defined for whole numbers 0 and up; square and cube still work for anything else
    if number < 0 or number != int(number):
        factorial = "Not defined: factorial needs a whole number of 0 or more"
    elif number > MAX_FACTORIAL:
        factorial = f"Too large to show: please use a number up to {MAX_FACTORIAL}"
    else:
        factorial = str(math.factorial(int(number)))

    # Results go out as text, not numbers: returning a dictionary to gr.JSON fails from 21! upwards
    # ("Integer exceeds 64-bit range"), because factorials quickly outgrow 64-bit integers
    return format_number(square), format_number(cube), factorial


def format_number(x):
    if x == int(x):
        return str(int(x))
    return f"{x:.10g}"


demo = gr.Interface(
    fn=superpowers,
    inputs=gr.Number(label="Your number", value=5),
    outputs=[
        gr.Textbox(label="Square (n × n)"),
        gr.Textbox(label="Cube (n × n × n)"),
        gr.Textbox(label="Factorial (1 × 2 × ... × n)", lines=3),
    ],
    title="Number Superpowers",
    description="Enter a number to get its square, cube and factorial.",
    examples=[[5], [12], [25], [-3], [2.5]],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
