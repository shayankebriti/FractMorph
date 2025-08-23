import os
from datetime import datetime

def save_plot(fig, method_name, folder='./artifacts/plots'):
    os.makedirs(folder, exist_ok=True)
    date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{method_name}_{date_time}.png"
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath)
