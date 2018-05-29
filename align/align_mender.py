# coding=utf-8
import pandas as pd
from align.align_tools import load_boundaries


class AlignMender(object):
    @staticmethod
    def mend(aligns_dict, predictions, bound_info):
        """
        Mend aligns by input params.
        :param aligns_dict:
        :param predictions:
        :param bound_info:
            list: [(align_name, bound_index, time), ...]
        :return:
            new_align_dict: a mended dict correspond to align_dict
            bound_dict:
            bound_count:
            bound_moved:
            move_dist_mean: move_dist_sum/bound_moved
        """
        wav_names, bound_indices, times = zip(*bound_info)
        df = pd.DataFrame({'wav_names': wav_names, 'bound_indices': bound_indices,
                           'times': times, 'predictions': predictions})
        bound_dict = load_boundaries(aligns_dict)

        bound_count = 0
        bound_moved = 0
        move_dist_sum = 0

        for (name, idx), group in df[['predictions', 'times']].groupby([wav_names, bound_indices]):
            preds = list(group.iloc[:, 0])
            assert len(preds) == 3
            '''judge three predictions, decide new boundary time and frame distance'''
            old_time, last_phone, next_phone, old_frame_dist = bound_dict[name][idx]
            '''make new boundaries'''
            new_time, new_frame_dist, moved, move_dist = AlignMender.__update_boundary(preds, old_frame_dist, old_time)
            bound_dict[name][idx] = (new_time, last_phone, next_phone, new_frame_dist)
            '''statistic move info'''
            if moved:
                bound_moved += 1
                move_dist_sum += move_dist
            bound_count += 1
        move_dist_mean = move_dist_sum/bound_moved if bound_moved != 0 else 0

        '''refresh boundaries of align_dict'''
        new_align_dict = AlignMender.__apply_boundaries(aligns_dict, bound_dict)
        return new_align_dict, bound_dict, bound_count, bound_moved, move_dist_mean

    @staticmethod
    def __update_boundary(preds, old_frame_dist, old_time, fs=16000):
        """
        Judge three predictions, decide new boundary time and frame distance
        :param preds:
        :param old_frame_dist:
        :param old_time:
        :param fs: frequency of sampling
        :return:
            new_time,
            new_frame_dist
            moved,
            move_dist
        """
        assert len(preds) == 3
        new_frame_dist = old_frame_dist
        new_time = old_time
        moved = False
        move_dist = None

        func_map = {
            '0-0-0': lambda t, d: (t+2*d/fs, d),
            '0-0-1': lambda t, d: (t+d/fs, d),
            '0-0-2': lambda t, d: (t+d/(fs*2), d/2),
            '0-1-2': lambda t, d: (t, d/2),
            '0-2-2': lambda t, d: (t-d/(fs*2), d/2),
            '1-2-2': lambda t, d: (t-d/fs, d),
            '2-2-2': lambda t, d: (t-d/(fs*2), d),
        }
        key = '{}-{}-{}'.format(*preds)
        if key in func_map.keys():
            new_time, new_frame_dist = func_map[key](old_time, old_frame_dist)
            moved = True
            move_dist = new_time-old_time
        return new_time, new_frame_dist, moved, move_dist

    @staticmethod
    def __apply_boundaries(aligns_dict, bound_dict):
        for k in aligns_dict.keys():
            aligns_dict[k].set_boundaries(bound_dict[k])
            # aligns_dict[k] = aligns_dict[k].set_boundaries(bound_dict[k])
        return aligns_dict
