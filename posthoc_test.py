from scipy import stats
import numpy as np

# zero_shot = [38.45,34.84,36.68,31.17,26.49,19.66,19.46,18.91,53.97,44.71,43.00,26.45,46.63,40.76,45.13,27.15,39.41,37.55,37.40,34.67]
# basic_combined = [39.31,38.77,36.43,37.19,30.81,28.85,19.93,19.73,51.96,44.92,36.56,48.78,48.50,44.56,42.17,45.97,39.09,37.18,38.28,37.93]
# basic_apidoc = [38.72,35.05,33.83,36.35,54.95,45.87,26.25,41.15,47.54,36.25,36.25,36.76,47.54,36.25,36.25,36.76,40.55,38.57,36.60,37.56]
# basic_issues = [39.50,36.25,36.71,38.29,31.11,19.73,19.66,18.89,50.20,44.51,35.36,46.24,49.90,44.98,40.56,47.78,39.43,37.74,39.61,38.09]
# basic_sos = [37.49,35.54,34.67,36.26,26.81,26.81,26.53,19.78,49.71,46.65,36.16,48.76,48.58,44.56,42.00,43.95,39.60,38.00,38.13,40.57]
# api_combined = [40.62,37.56,34.96,38.39,26.99,26.27,19.14,29.17,56.23,47.21,42.07,46.31,48.86,41.34,43.49,43.67,41.37,39.32,38.77,38.62]
# api_apidoc = [38.46,37.04,35.94,36.49,20.15,19.78,20.02,20.09,54.62,44.12,39.40,44.81,50.18,44.16,35.88,40.91,40.28,38.01,37.37,37.64]
# api_issues = [40.19,38.35,37.09,39.17,30.65,30.59,19.72,31.18,49.59,46.45,43.10,46.67,54.26,47.68,43.42,49.53,40.39,39.01,38.48,39.27]
# api_sos = [39.32,36.57,35.73,36.68,26.54,26.59,19.31,26.49,53.45,45.20,32.69,47.20,46.89,44.84,42.07,44.14,39.81,38.94,38.29,40.66]

zero_shot = [39.08, 36.72, 37.05, 32.84, 25.66, 25.43, 18.29, 17.43, 68.19, 59.76, 58.49, 22.78, 56.93, 48.12, 51.52, 16.72, 34.45, 33.14, 33.01, 30.95]
basic_combined = [39.12, 37.95, 37.68, 38.22, 31.21, 30.95, 31.21, 25.40, 66.52, 56.35, 49.32, 50.03, 56.06, 45.61, 41.35, 54.27, 34.59, 32.80, 33.53, 34.24]
basic_apidoc = [39.25, 36.89, 37.03, 38.94, 18.77, 18.09, 17.49, 18.24, 69.52, 55.89, 42.72, 54.77, 57.17, 49.65, 42.74, 44.52, 35.03, 34.02, 31.79, 33.70]
basic_issues = [39.28, 37.22, 37.51, 38.66, 31.31, 30.92, 30.82, 17.46, 67.15, 44.51, 49.94, 47.39, 55.42, 44.80, 44.61, 53.70, 34.66, 33.66, 35.04, 34.38]
basic_sos = [39.35, 35.96, 35.64, 38.82, 25.83, 25.62, 25.47, 26.03, 64.94, 53.68, 47.64, 61.33, 55.93, 41.56, 50.10, 53.34, 34.79, 33.50, 32.88, 35.77]
api_combined = [40.09, 38.91, 37.16, 39.31, 26.38, 31.27, 30.79, 32.16, 69.54, 22.28, 55.88, 18.73, 58.71, 36.23, 47.57, 36.23, 35.88, 34.52, 34.30, 34.21]
api_apidoc = [39.37, 37.47, 36.97, 38.10, 19.18, 25.50, 25.78, 18.96, 69.82, 21.67, 55.57, 19.85, 57.65, 44.11, 44.81, 29.97, 34.92, 33.41, 33.54, 33.29]
api_issues = [40.28, 38.53, 37.78, 38.94, 31.89, 33.21, 32.35, 32.90, 49.59, 22.11, 51.79, 19.32, 57.73, 43.59, 47.09, 41.65, 35.33, 34.59, 35.12, 34.84]
api_sos = [38.83, 37.22, 36.50, 38.93, 25.76, 25.90, 25.50, 26.01, 66.15, 96.45, 57.41, 20.82, 55.19, 32.88, 45.74, 35.66, 35.98, 36.61, 36.19, 37.26]


# perform Friedman Test
freidman_basic = stats.friedmanchisquare(
    zero_shot,
    basic_combined,
    basic_apidoc,
    basic_issues,
    basic_sos,
)

freidman_api = stats.friedmanchisquare(
    zero_shot,
    api_combined,
    api_apidoc,
    api_issues,
    api_sos
)

freidman_all = stats.friedmanchisquare(
    zero_shot,
    basic_combined,
    basic_apidoc,
    basic_issues,
    basic_sos,
    api_combined,
    api_apidoc,
    api_issues,
    api_sos
)

basic_rag = np.array([
    zero_shot,
    basic_combined,
    basic_apidoc,
    basic_issues,
    basic_sos,
]).T

api_rag = np.array([
    zero_shot,
    api_combined,
    api_apidoc,
    api_issues,
    api_sos
]).T

all_rags = np.array([
    zero_shot,
    basic_combined,
    basic_apidoc,
    basic_issues,
    basic_sos,
    api_combined,
    api_apidoc,
    api_issues,
    api_sos
]).T

basic_ranks = np.apply_along_axis(lambda x: stats.rankdata(x, method='min'), 1, basic_rag)
api_ranks = np.apply_along_axis(lambda x: stats.rankdata(x, method='min'), 1, api_rag)
all_ranks = np.apply_along_axis(lambda x: stats.rankdata(x, method='min'), 1, all_rags)

reversed_basic_ranks = np.max(basic_ranks, axis=1, keepdims=True) - basic_ranks + 1
reversed_api_ranks = np.max(api_ranks, axis=1, keepdims=True) - api_ranks + 1
reversed_all_ranks = np.max(all_ranks, axis=1, keepdims=True) - all_ranks + 1

average_basic_ranks = np.mean(reversed_basic_ranks, axis=0)
average_api_ranks = np.mean(reversed_api_ranks, axis=0)
average_all_ranks = np.mean(reversed_all_ranks, axis=0)

basic_group_labesl = ['zeroshot', 'basic combined', 'basic apidoc', 'basic issues', 'basic sos']
api_group_labels = ['zeroshot', 'api combined', 'api apidoc', 'api issues', 'api sos']
all_group_labels = ['zeroshot', 'basic combined', 'basic apidoc', 'basic issues', 'basic sos', 'api combined', 'api apidoc', 'api issues', 'api sos']

basic_rankings = dict(zip(basic_group_labesl, average_basic_ranks))
api_rankings = dict(zip(api_group_labels, average_api_ranks))
all_rankings = dict(zip(all_group_labels, average_all_ranks))

# Sort the dictionary by average rank
sorted_basic_rankings = dict(sorted(basic_rankings.items(), key=lambda item: item[1]))
sorted_api_rankings = dict(sorted(api_rankings.items(), key=lambda item: item[1]))
sorted_all_rankings = dict(sorted(all_rankings.items(), key=lambda item: item[1]))


print(f'Friedman for basic RAG: {freidman_basic}')
print(f'Friedman for api RAG: {freidman_api}')
print(f'Friedman for all RAG: {freidman_all}')
print(f"Average Ranks for basic RAG: {average_basic_ranks}")
print(f"Average Ranks for api RAG: {average_api_ranks}")
print(f"Average Ranks for all RAG: {average_all_ranks}")
print(f"Sorted Rankings for basic RAG: {sorted_basic_rankings}")
print(f"Sorted Rankings for api RAG: {sorted_api_rankings}")
print(f"Sorted Rankings for all RAG: {sorted_all_rankings}")
