import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import csv

def save_dqn_parameters(results_folder, lr, gamma, eps, eps_min, batch_size, mem_size, min_mem_size, n_step, target_update, model_flag):

    csv_path = os.path.join(results_folder,'parameters.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['learning rate',lr])
            csvwriter.writerow(['gamma',gamma])
            csvwriter.writerow(['epsilon',eps])
            csvwriter.writerow(['epsilon min',eps_min])
            csvwriter.writerow(['batch size',batch_size])
            csvwriter.writerow(['mem size',mem_size])
            csvwriter.writerow(['min memory for training',min_mem_size])
            csvwriter.writerow(['n_step',n_step])
            csvwriter.writerow(['target update',target_update])
            csvwriter.writerow(['model flag',model_flag])


def print_board(self, observation, flag_x, results_folder):

    (grid, _, possibilities) = observation
    with open(os.path.join(results_folder,"board.txt"), 'a') as file:
        file.write('\n')
        file.write('X' if flag_x else 'O')
        file.write('\n')
        
        for i in range(9):
            print(*grid[i*9:i*9+9],file=file)
        file.write("----------------\n")
        file.write("possibilities:")
        print(*possibilities, sep=' ',file=file)
        file.write('\n')
        file.write("----------------\n")
    return


def save_loss(loss, loss_arr, results_folder):

    loss_avg = torch.mean(torch.tensor(loss_arr))
    loss.append(loss_avg.item())
    print('\n',loss_avg)

    with open(os.path.join(results_folder,'loss.csv'), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(loss_arr)

    return loss


def save_loss_graph(results_folder, loss, episode): 

    output_dir = os.path.join(results_folder,'loss_graphs')
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(loss)
    plt.xlabel('# episodes [x 50]')
    plt.ylabel('loss')
    plt.title(f'Average loss')
    plt.savefig(os.path.join(output_dir, f'loss_epoch_{episode + 1}.png'))
    plt.close()


def save_results_against_random(results_folder, episode, results1, results2 = []):

    with open(os.path.join(results_folder,'results.csv'), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([episode + 1, 1, *results1])
        if results2:
            csvwriter.writerow([episode + 1, 2, *results2])


    