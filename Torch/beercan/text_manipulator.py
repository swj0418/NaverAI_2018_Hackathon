
#######################################################################################################################
#                                       GUIDELINES
# 1. All characters are referred to as "x"
#
#######################################################################################################################


class KorTex:
    # For Standard Decomposition
    KOR_INIT_START = "ㄱ/ㄲ/ㄴ/ㄷ/ㄸ/ㄹ/ㅁ/ㅂ/ㅃ/ㅅ/ㅆ/ㅇ/ㅈ/ㅉ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split('/')  # len = 19
    KOR_INIT_MIDDLE = "ㅏ/ㅐ/ㅑ/ㅒ/ㅓ/ㅔ/ㅕ/ㅖ/ㅗ/ㅘ/ㅙ/ㅚ/ㅛ/ㅜ/ㅝ/ㅞ/ㅟ/ㅠ/ㅡ/ㅢ/ㅣ".split('/')  # len = 21
    # len = 27
    KOR_INIT_END = "0/ㄱ/ㄲ/ㄳ/ㄴ/ㄵ/ㄶ/ㄷ/ㄹ/ㄺ/ㄻ/ㄼ/ㄽ/ㄾ/ㄿ/ㅀ/ㅁ/ㅂ/ㅄ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split(
        '/')
    KOR_INIT_END_N = "ㄱ/ㄲ/ㄳ/ㄴ/ㄵ/ㄶ/ㄷ/ㄹ/ㄺ/ㄻ/ㄼ/ㄽ/ㄾ/ㄿ/ㅀ/ㅁ/ㅂ/ㅄ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split(
        '/')

    # For Total Decomposition
    KOR_ATOM_START = ""

    test = KOR_INIT_START + KOR_INIT_MIDDLE + KOR_INIT_END

    hangul_length = len(KOR_INIT_START) + len(KOR_INIT_MIDDLE) + len(KOR_INIT_END_N)

    def __init__(self, x: str):
        self.x = x

    def is_valid_decompostion_atom(self):
        return self.x in self.test

    def decompose_initial_sound(self):
        if self.x is None:
            print("Exception from 'decompose_initial_sound' REASON : No text was given")
            return None;

        x_strip = str.strip(self.x)
        initial_sound_stripped = []

        for idx in range(len(x_strip)):
            char = x_strip[idx]
            if ord(char) >= ord('가') and ord(char) <= ord('힣'):
                initial_sound_stripped.append(self.KOR_INIT_START[int((ord(char) - ord('가')) / (21 * 28))])

        return initial_sound_stripped

    # 뜸들여 to ㄸ ㅡ ㅁ ㄷ ㅡ ㄹ ㅇ ㅕ
    def decompose_light(self):
        if self.x is None:
            print("Exception from 'decompose_light' REASON : No text was given")
            return None;

        # Maximum length of 3 can be allocated per letter

        x_strip = str.strip(self.x)
        lightly_decomposed = []
        for idx in range(len(x_strip)):
            begin_point = ord(x_strip[idx]) - ord('가')

            if ord(x_strip[idx]) >= ord('가') and ord(x_strip[idx]) <= ord('힣'):
                # Initial Sound
                lightly_decomposed.append(self.KOR_INIT_START[int(begin_point / (21 * 28))])
                # Middle Sound
                begin_point = int((begin_point % (21 * 28)))
                lightly_decomposed.append(self.KOR_INIT_MIDDLE[int(begin_point / 28)])
                # End Sound
                begin_point = int(begin_point % 28)

                if begin_point % 28 is not 0:
                    lightly_decomposed.append(self.KOR_INIT_END[begin_point])
            else:
                lightly_decomposed.append(x_strip[idx])

        return lightly_decomposed

    def decompose_heavy(self):
        if self.x is None:
            return None

    def word2int_raw(self):
        decomposed: int = []

        x_strip = str.strip(self.x)
        for idx in range(len(x_strip)):
            decomposed.append(ord(x_strip[idx]) - ord('가'))

        return decomposed

    def __word2int_ordinal(self, x, warning=True):
        one_hot = []

        if ord('가') <= x <= ord('힣'):
            x = x - ord('가')
            y = x // 28
            z = x % 28
            x = y // 21
            y = y % 21

            zz = self.KOR_INIT_END_N[z - 1] if z > 0 else ''
            if x >= len(self.KOR_INIT_START):
                if warning:
                    print('Unknown Exception: ', x,
                          chr(x), x, y, z, zz)

            one_hot.append(x)
            one_hot.append(len(self.KOR_INIT_START) + y)
            if z > 0:
                one_hot.append(len(self.KOR_INIT_START) + len(self.KOR_INIT_MIDDLE) + (z - 1))
            return one_hot

        else:
            if x < 128:
                result = self.hangul_length + x  # 67~
            elif ord('ㄱ') <= x <= ord('ㅣ'):
                # 194~ # [ㄱ:12593]~[ㅣ:12643] (len = 51)
                result = self.hangul_length + 128 + (x - 12593)
            elif x == ord('♡'):
                result = self.hangul_length + 128 + 51  # 245~ # ♡
            elif x == ord('♥'):
                result = self.hangul_length + 128 + 51 + 1  # ♥
            elif x == ord('★'):
                result = self.hangul_length + 128 + 51 + 2  # ★
            elif x == ord('☆'):
                result = self.hangul_length + 128 + 51 + 3  # ☆
            else:
                if warning:
                    print('Unhandled character:', chr(x), x)
                    #Unknown Character
                result = self.hangul_length + 128 + 51 + 4  # for unknown character

            return [result]

    def word2int_ordinal(self, warning=True):
        decomposed = []
        x_strip = str.strip(self.x)

        for x in x_strip:
            decomped_word = self.__word2int_ordinal(ord(x), warning=warning)
            decomposed.extend(decomped_word)

        return decomposed



KT = KorTex("안녕하세요")
D1 = KT.decompose_initial_sound()
D2 = KT.decompose_light()
print(D1)
print(D2)

KT2 = KorTex("가나다라마바사")
D1 = KT2.word2int_raw()
print(D1)

KT3 = KorTex("나는 오늘 5시까지 공부를 헀다. 하지만 더 할 것이다.")
D0 = KT3.word2int_raw()
D1 = KT3.word2int_ordinal()
print(D0)
print(D1)

KT4 = KorTex("없꼢")
print(KT4.word2int_ordinal())
print(len(KT.KOR_INIT_START))
print(len(KT.KOR_INIT_MIDDLE))
print(len(KT.KOR_INIT_END_N
          ))