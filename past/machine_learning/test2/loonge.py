import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x_data = np.linspace(-100, 100, 1000)


def f(x):
    return 1 / (1 + 25 * x ** 2)


y_true = f(x_data)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = y_true + noise


degrees = [3, 5, 7, 9, 11]
models = {}
errors = {}

plt.figure(figsize=(14, 8))

plt.scatter(x_data, y_data, s=5, alpha=0.5, label='Noisy Sample Points', color='gray')

for deg in degrees:
    coeffs = np.polyfit(x_data, y_data, deg)
    poly_func = np.poly1d(coeffs)
    models[deg] = poly_func

    y_pred_values = poly_func(x_data)

    sse = np.sum((y_data - y_pred_values) ** 2)
    errors[deg] = sse
    plt.plot(x_data, y_pred_values, label=f'Degree {deg} (SSE={sse:.2e})', linewidth=2)




# plt.title(r'Polynomial Regression on $f(x)=\frac{1}{1+25x^2}$', fontsize=16)
# plt.xlabel('x', fontsize=12)
# plt.ylabel('y', fontsize=12)
# plt.legend(fontsize=10, loc='upper right')
#
# plt.ylim(-2, 2)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()




for deg in degrees:
    sse = errors[deg]
    if sse > 1e10:
        note = "严重发散"
    elif sse > 100:
        note = "过拟合"
    else:
        note = "相对平稳"
    print(f"{deg:<10} {sse:<25.4e} {note:<15}")




















