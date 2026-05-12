import re
import math
import numpy as np
from ._masker import Masker
from ..utils import safe_isinstance
from ..utils.transformers import parse_prefix_suffix_for_tokenizer, SENTENCEPIECE_TOKENIZERS, getattr_silent


class Text(Masker):
    """ This masks out tokens according to the given tokenizer.

    The masked variables are

    output_type : "string" (default) or "token_ids"
    """
    def __init__(self, tokenizer=None, mask_token=None, collapse_mask_token="auto", output_type="string"):
        """ Build a new Text masker given an optional passed tokenizer.

        Parameters
        ----------
        tokenizer : callable or None
            The tokenizer used to break apart strings during masking.

        mask_token : string, int, or None
            The sub-string or integer token id used to mask out portions of a string.

        collapse_mask_token : True, False, or "auto"
            If True, when several consecutive tokens are masked only one mask token is used.
        """

        if tokenizer is None:
            self.tokenizer = SimpleTokenizer()
        elif callable(tokenizer):
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = SimpleTokenizer(tokenizer)
            except:
                raise Exception(  # pylint: disable=raise-missing-from
                    "The passed tokenizer cannot be wrapped as a masker because it does not have a __call__ "
                    "method, not can it be interpreted as a splitting regexp!"
                )

        self.output_type = output_type
        self.collapse_mask_token = collapse_mask_token
        self.input_mask_token = mask_token
        self.mask_token = mask_token
        self.mask_token_id = mask_token if isinstance(mask_token, int) else None
        parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(self.tokenizer)

        self.keep_prefix = parsed_tokenizer_dict['keep_prefix']
        self.keep_suffix = parsed_tokenizer_dict['keep_suffix']

        self.text_data = True

        if mask_token is None:
            if getattr_silent(self.tokenizer, "mask_token") is not None:
                self.mask_token = self.tokenizer.mask_token
                self.mask_token_id = getattr_silent(self.tokenizer, "mask_token_id")
                if self.collapse_mask_token == "auto":
                    self.collapse_mask_token = False
            else:
                self.mask_token = "..."
        else:
            self.mask_token = mask_token

        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer(self.mask_token)["input_ids"][self.keep_prefix]

        if self.collapse_mask_token == "auto":
            self.collapse_mask_token = True

        # note if this masker can use a different background for different samples
        self.fixed_background = self.mask_token_id is None

        self.default_batch_size = 5

        # cache variables
        self._s = None
        self._tokenized_s_full = None
        self._tokenized_s = None
        self._segments_s = None

        # flag that we return outputs that will not get changed by later masking calls
        self.immutable_outputs = True

    def __call__(self, mask, s):
        mask = self._standardize_mask(mask, s)
        self._update_s_cache(s)

        # if we have a fixed prefix or suffix then we need to grow the mask to account for that
        if self.keep_prefix > 0 or self.keep_suffix > 0:
            mask = mask.copy()
            if self.keep_prefix > 0:
                mask[:self.keep_prefix] = True
            if self.keep_suffix > 0:
                mask[-self.keep_suffix:] = True

        if self.output_type == "string":
            out_parts = []
            is_previous_appended_token_mask_token = False
            sep_token = getattr_silent(self.tokenizer, "sep_token")
            for i, v in enumerate(mask):
                # mask ignores separator tokens and keeps them unmasked
                if v or sep_token == self._segments_s[i]:
                    out_parts.append(self._segments_s[i])
                    is_previous_appended_token_mask_token = False
                else:
                    if not self.collapse_mask_token or (self.collapse_mask_token and not is_previous_appended_token_mask_token):
                        out_parts.append(" " + self.mask_token)
                        is_previous_appended_token_mask_token = True
            out = "".join(out_parts)

            # replace whitespace encoded characters for sentencepiece tokenizers
            if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
                out = out.replace('\u2581', ' ')

            # replace sequence of spaces with a single space and strip
            out = re.sub(r"[\s]+", " ", out).strip()

        else:
            if self.mask_token_id is None:
                out = self._tokenized_s[mask]
            else:
                out = np.array([self._tokenized_s[i] if mask[i] else self.mask_token_id for i in range(len(mask))])

        return (np.array([out]),)

    def data_transform(self, s):
        """ Called by explainers to allow us to convert data to better match masking (here this means tokenizing).
        """
        return (self.token_segments(s)[0],)

    def token_segments(self, s):
        """ Returns the substrings associated with each token in the given string.
        """

        try:
            token_data = self.tokenizer(s, return_offsets_mapping=True)
            offsets = token_data["offset_mapping"]
            offsets = [(0, 0) if o is None else o for o in offsets]
            parts = [s[offsets[i][0]:max(offsets[i][1], offsets[i+1][0])] for i in range(len(offsets)-1)]
            parts.append(s[offsets[len(offsets)-1][0]:offsets[len(offsets)-1][1]])
            return parts, token_data["input_ids"]
        except (NotImplementedError, TypeError):
            token_ids = self.tokenizer(s)['input_ids']
            tokens = [self.tokenizer.decode([id]) for id in token_ids]
            if hasattr(self.tokenizer, "get_special_tokens_mask"):
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
                special_keep = [getattr_silent(self.tokenizer, 'sep_token'), getattr_silent(self.tokenizer, 'mask_token')]
                for i, v in enumerate(special_tokens_mask):
                    if v == 1 and (tokens[i] not in special_keep or i + 1 == len(special_tokens_mask)):
                        tokens[i] = ""

            # add spaces to separate the tokens
            if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
                for i, v in enumerate(tokens):
                    if v.startswith("_"):
                        tokens[i] = " " + tokens[i][1:]
            else:
                for i, v in enumerate(tokens):
                    if v.startswith("##"):
                        tokens[i] = tokens[i][2:]
                    elif v != "" and i != 0:
                        tokens[i] = " " + tokens[i]

            return tokens, token_ids

    def clustering(self, s):
        """ Compute the clustering of tokens for the given string.
        """
        self._update_s_cache(s)
        special_tokens = []
        sep_token = getattr_silent(self.tokenizer, "sep_token")
        if sep_token is None:
            special_tokens = []
        else:
            special_tokens = [sep_token]

        # convert the text segments to tokens that the partition tree function expects
        tokens = []
        space_end = re.compile(r"^.*\W$")
        letter_start = re.compile(r"^[A-z]")
        for i, v in enumerate(self._segments_s):
            if i > 0 and space_end.match(self._segments_s[i-1]) is None and letter_start.match(v) is not None and tokens[i-1] != "":
                tokens.append("##" + v.strip())
            else:
                tokens.append(v.strip())

        pt = partition_tree(tokens, special_tokens)

        # use the rescaled size of the clusters as their height
        pt[:, 2] = pt[:, 3]
        pt[:, 2] /= pt[:, 2].max()

        return pt

    def _update_s_cache(self, s):
        if self._s != s:
            self._s = s
            tokens, token_ids = self.token_segments(s)
            self._tokenized_s = np.array(token_ids)
            self._segments_s = np.array(tokens)

    def shape(self, s):
        """ The shape of what we return as a masker.

        Note we only return a single sample, so there is no expectation averaging.
        """
        self._update_s_cache(s)
        return (1, len(self._tokenized_s))

    def mask_shapes(self, s):
        """ The shape of the masks we expect.
        """
        self._update_s_cache(s)
        return [(len(self._tokenized_s),)]

    def invariants(self, s):
        """ The names of the features for each mask position for the given input string.
        """
        self._update_s_cache(s)

        invariants = np.zeros(len(self._tokenized_s), dtype=bool)
        if self.keep_prefix > 0:
            invariants[:self.keep_prefix] = True
        if self.keep_suffix > 0:
            invariants[-self.keep_suffix:] = True
        # mark separator tokens as invariant
        for i, v in enumerate(self._tokenized_s):
            if v == getattr_silent(self.tokenizer, "sep_token_id"):
                invariants[i] = True
        return invariants.reshape(1, -1)

    def feature_names(self, s):
        """ The names of the features for each mask position for the given input string.
        """
        self._update_s_cache(s)
        return [[v.strip() for v in self._segments_s]]


class SimpleTokenizer():  # pylint: disable=too-few-public-methods
    """ A basic model agnostic tokenizer.
    """
    def __init__(self, split_pattern=r"\W+"):
        """ Create a tokenizer based on a simple splitting pattern.
        """
        self.split_pattern = re.compile(split_pattern)

    def __call__(self, s, return_offsets_mapping=True):
        """ Tokenize the passed string, optionally returning the offsets of each token in the original string.
        """
        pos = 0
        offset_ranges = []
        input_ids = []
        for m in re.finditer(self.split_pattern, s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])

        out = {}
        out["input_ids"] = input_ids
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        return out


def post_process_sentencepiece_tokenizer_output(s):
    """ replaces whitespace encoded as '_' with ' ' for sentencepiece tokenizers.
    """
    s = s.replace('\u2581', ' ')
    return s

openers = {
    "(": ")"
}
closers = {
    ")": "("
}
enders = [".", ","]
connectors = ["but", "and", "or"]


class Token():
    """ A token representation used for token clustering.
    """
    def __init__(self, value):
        self.s = value
        if value in openers or value in closers:
            self.balanced = False
        else:
            self.balanced = True

    def __str__(self):
        return self.s

    def __repr__(self):
        if not self.balanced:
            return self.s + "!"
        return self.s


class TokenGroup():
    """ A token group (substring) representation used for token clustering.
    """
    def __init__(self, group, index=None):
        self.g = group
        self.index = index

    def __repr__(self):
        return self.g.__repr__()

    def __getitem__(self, index):
        return self.g[index]

    def __add__(self, o):
        return TokenGroup(self.g + o.g)

    def __len__(self):
        return len(self.g)


def merge_score(group1, group2, special_tokens):
    """ Compute the score of merging two token groups.

    special_tokens: tokens (such as separator tokens) that should be grouped last
    """
    score = 0
    # ensures special tokens are combined last
    if len(special_tokens) > 0:
        if group1[-1].s in special_tokens and group2[0].s in special_tokens:
            score -= math.inf

    # merge broken-up parts of words first
    if group2[0].s.startswith("##"):
        score += 20

    # merge apostrophe endings next
    if group2[0].s == "'" and (len(group2) == 1 or (len(group2) == 2 and group2[1].s in ["t", "s"])):
        score += 15
    if group1[-1].s == "'" and group2[0].s in ["t", "s"]:
        score += 15

    start_ctrl = group1[0].s.startswith("[") and group1[0].s.endswith("]")
    end_ctrl = group2[-1].s.startswith("[") and group2[-1].s.endswith("]")

    if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
        score -= 1000
    if group2[0].s in openers and not group2[0].balanced:
        score -= 100
    if group1[-1].s in closers and not group1[-1].balanced:
        score -= 100

    # attach surrounding an openers and closers a bit later
    if group1[0].s in openers and not group2[-1] in closers:
        score -= 2

    # reach across connectors later
    if group1[-1].s in connectors or group2[0].s in connectors:
        score -= 2

    # reach across commas later
    if group1[-1].s == ",":
        score -= 10
    if group2[0].s == ",":
        if len(group2) > 1:
            score -= 10
        else:
            score -= 1

    # reach across sentence endings later
    if group1[-1].s in [".", "?", "!"]:
        score -= 20
    if group2[0].s in [".", "?", "!"]:
        if len(group2) > 1:
            score -= 20
        else:
            score -= 1

    score -= len(group1) + len(group2)
    return score


def merge_closest_groups(groups, special_tokens):
    """ Finds the two token groups with the best merge score and merges them.
    """
    scores = [merge_score(groups[i], groups[i+1], special_tokens) for i in range(len(groups)-1)]
    ind = np.argmax(scores)
    groups[ind] = groups[ind] + groups[ind+1]
    if groups[ind][0].s in openers and groups[ind+1][-1].s == openers[groups[ind][0].s]:
        groups[ind][0].balanced = True
        groups[ind+1][-1].balanced = True

    groups.pop(ind+1)


def partition_tree(decoded_tokens, special_tokens):
    """ Build a hierarchical clustering of tokens that align with sentence structure.

    Note that this is fast and heuristic right now.
    """
    token_groups = [TokenGroup([Token(t)], i) for i, t in enumerate(decoded_tokens)]
    M = len(decoded_tokens)
    new_index = M
    clustm = np.zeros((M-1, 4))
    for i in range(len(token_groups)-1):
        scores = [merge_score(token_groups[i], token_groups[i+1], special_tokens) for i in range(len(token_groups)-1)]
        ind = np.argmax(scores)

        lind = token_groups[ind].index
        rind = token_groups[ind+1].index
        clustm[new_index-M, 0] = token_groups[ind].index
        clustm[new_index-M, 1] = token_groups[ind+1].index
        clustm[new_index-M, 2] = -scores[ind]
        clustm[new_index-M, 3] = (clustm[lind-M, 3] if lind >= M else 1) + (clustm[rind-M, 3] if rind >= M else 1)

        token_groups[ind] = token_groups[ind] + token_groups[ind+1]
        token_groups[ind].index = new_index

        # track balancing of openers/closers
        if token_groups[ind][0].s in openers and token_groups[ind+1][-1].s == openers[token_groups[ind][0].s]:
            token_groups[ind][0].balanced = True
            token_groups[ind+1][-1].balanced = True

        token_groups.pop(ind+1)
        new_index += 1

    # negative means we should never split a group
    clustm[:, 2] = clustm[:, 2] + 10

    return clustm
