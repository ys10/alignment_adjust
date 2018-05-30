# coding=utf-8


class Align(object):
    suffix = '.'

    def __init__(self, default_distance=320):
        self.name = None
        self.phones = list()  # List of list (phoneme, begin_time, end_time).
        self.boundaries = None
        self.frame_distance = default_distance  # Default frame distance(number of sample).

    def to_lines(self):
        lines = list()
        lines.append(self.name + '\n')
        for p in self.phones:
            line = '{}\t[{},{}]\n'.format(p[0], p[1], p[2])
            lines.append(line)
        lines.append(self.suffix + '\n')
        return lines

    def get_boundaries(self):
        """
        Get boundary list of current alignment.
        :return boundaries:
            list: [(time1, phone0, phone1, distance1), (time2, phone1, phone2, distance2), ...]
        """
        if self.boundaries is not None:
            return self.boundaries
        boundaries = list()
        last_end = None  # End time of last phoneme.
        last_phone = None  # Name of last phoneme.

        for p in self.phones:
            if last_end is None:  # First phoneme.
                last_end = float(p[2])
                last_phone = p[0]
            else:
                '''Calculate time boundary by mean of
                 last end time and current begin time.'''
                time = (last_end + float(p[1])) / 2
                '''Boundary tuple:
                 (boundary_time, last_phoneme, current_phoneme)'''
                boundaries.append((time, last_phone, p[0], self.frame_distance))
                '''Set new end time & phoneme name'''
                last_end = float(p[2])
                last_phone = p[0]

        return boundaries

    def set_boundaries(self, boundaries):
        self.boundaries = boundaries
        if len(self.phones) != len(boundaries)+1:
            return False

        for i in range(len(boundaries)):
            self.phones[i][2] = boundaries[i][0]
            self.phones[i+1][1] = boundaries[i][0]
        return True


def __is_integer(num):
    try:
        int(num)
        return True
    except ValueError:
        return False


def load_aligns(align_path):
    aligns = dict()  # Dict of Align instances.

    with open(align_path, 'r') as fin:
        align = None
        while 1:
            line = fin.readline()
            if not line:  # End of file.
                break
            '''Not end'''
            text = line.strip()

            if __is_integer(text):  # Align begin (name).
                align = Align()
                align.name = text
            elif text == Align.suffix or text == '':  # Align end.
                if align:
                    aligns[align.name] = align
                    align = None
            else:  # Align phones.
                phone, time = text.split(maxsplit=1)
                begin, end = time[1:-1].split(',')
                align.phones.append([phone, begin, end])

    return aligns


def save_aligns(aligns_dict, save_path):
    aligns_list = list()
    for k in aligns_dict:
        aligns_list.append(aligns_dict[k])
    aligns_list.sort(key=lambda x: x.name)

    with open(save_path, 'w', newline='\n') as f:
        for align in aligns_list:
            f.writelines(align.to_lines())


def load_boundaries(aligns):
    """
    Get boundaries for each alignment in aligns.
    :param aligns:
        dict {name1: Align1, name2: Align2, ...}
    :return boundary_dict:
        dict {name1: boundary_list1, name2: boundary_list2, ...}
    """
    boundary_dict = dict()
    for name in aligns.keys():
        boundary_dict[name] = aligns[name].get_boundaries()
    return boundary_dict
