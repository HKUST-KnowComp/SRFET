
SRL_SAFE_TOKENS = {'the', 'to', 'from'}


def get_srl_tag_spans(tag_seq):
    tag_spans = list()
    if 'B-V' not in tag_seq:
        return tag_spans

    i = 0
    seq_len = len(tag_seq)
    while i < len(tag_seq):
        if tag_seq[i].startswith('B'):
            pbeg, pend = i, i + 1
            while pend < seq_len and tag_seq[pend].startswith('I'):
                pend += 1
            i = pend
            tag_spans.append((tag_seq[pbeg][2:], (pbeg, pend)))
        else:
            i += 1
    return tag_spans


def __extra_tokens_ok(tokens, dep_tag_seq, beg_pos, end_pos, mention_span):
    for i in range(beg_pos, end_pos):
        if tokens[i].lower() in SRL_SAFE_TOKENS or (not tokens[i].isalpha()):
            continue

        dep_tag = dep_tag_seq[i]
        if dep_tag[0] == 'amod' and mention_span[0] <= dep_tag[1][0] < mention_span[1]:
            continue
        return False
    return True


def __missing_tokens_ok(tokens, beg_pos, end_pos):
    for i in range(beg_pos, end_pos):
        if tokens[i].lower() in SRL_SAFE_TOKENS or (not tokens[i].isalpha()):
            continue
        return False
    return True


def is_mention_match_with_dep(mspan, phrase_span, tokens, dep_tag_seq):
    mbeg, mend = mspan
    pbeg, pend = phrase_span
    if mbeg == pbeg and mend == pend:
        return True

    if mend <= pbeg or pend <= mbeg:
        return False

    if pbeg < mbeg:
        if not __extra_tokens_ok(tokens, dep_tag_seq, pbeg, mbeg, mspan):
            return False
    elif mbeg < pbeg:
        if not __missing_tokens_ok(tokens, mbeg, pbeg):
            return False

    if mend != pend:
        if not __missing_tokens_ok(tokens, min(mend, pend), max(mend, pend)):
            return False
    return True


def is_mention_match(mspan, phrase_span, tokens):
    mbeg, mend = mspan
    pbeg, pend = phrase_span
    if mbeg == pbeg and mend == pend:
        return True

    if abs(mbeg - pbeg) > 1 or abs(mend - pend) > 1:
        return False

    if abs(mbeg - pbeg) == 1:
        tleft = tokens[min(mbeg, pbeg)]
        if tleft.lower() not in SRL_SAFE_TOKENS and tleft.isalpha():
            return False

    if abs(mend - pend) == 1:
        tright = tokens[max(mend, pend) - 1]
        if tright.lower() not in SRL_SAFE_TOKENS and tright.isalpha():
            return False
    return True


def match_srl_to_mentions_all(sent_tokens, srl_results, mention_span, dep_tag_seq=None):
    pos_beg, pos_end = mention_span
    matched_tag_spans_list, matched_tag_list = list(), list()
    for tag_spans in srl_results:
        for tag, span in tag_spans:
            if dep_tag_seq is None:
                matched = is_mention_match((pos_beg, pos_end), span, sent_tokens)
            else:
                matched = is_mention_match_with_dep((pos_beg, pos_end), span, sent_tokens, dep_tag_seq)
            if matched and tag in {'ARG0', 'ARG1', 'ARG2'}:
                matched_tag_spans_list.append(tag_spans)
                matched_tag_list.append(tag)
    return matched_tag_list, matched_tag_spans_list


def get_srl_tag_span(srl_tag_spans, tag):
    for cur_tag, span in srl_tag_spans:
        if cur_tag == tag:
            return span
    return None
