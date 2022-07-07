class Seq2Seq(nn.Module):

    def __init__(
        self,
        input_size, # Source 언어의 단어 사이즈
        word_vec_size,
        hidden_size,
        output_size, # Target 언어의 단어 사이즈
        n_layers=4,
        dropout_p=.2
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x, length):
        # |x| = (bs, n)
        # |length| = (bs, ) / 미니배치내 각 샘플별 길이
        
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                    ], dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()

        return mask

    def merge_encoder_hiddens(self, encoder_hiddens): # 인코더 히든 스테이트를 받아서 디코더 히든 스테이트에 맞게끔 변환해주는 것
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens
        # |hiddens| = (n_layers*2, bs, hs/2 애초에 나누기 2를 해줬으니)
        
        # i-th and (i+1)-th layer is opposite direction.
        # Also, each direction of layer is half hidden size.
        # Therefore, we concatenate both directions to 1 hidden size layer.
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)] 
            # |new_hiddens| = (bs, hs/2 *2)
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)
        # |new_hiddens| = (n_layers, bs, hs)
        return (new_hiddens, new_cells)

    def fast_merge_encoder_hiddens(self, encoder_hiddens): # for문 사용 안하고 변환.
        # Merge bidirectional to uni-directional
        # We need to convert size from (n_layers * 2, batch_size, hidden_size / 2)
        # to (n_layers, batch_size, hidden_size).
        # Thus, the converting operation will not working with just 'view' method.
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        # contiguous는 메모리에 잘 붙어 있으라는 뜻
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        # You can use 'merge_encoder_hiddens' method, instead of using above 3 lines.
        # 'merge_encoder_hiddens' method works with non-parallel way.
        # h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        return h_0_tgt, c_0_tgt

    def forward(self, src, tgt):
        # |src| = (bs, n) = (bs, n, |V_src|)
        # |tgt| = (bs, m) = (bs, m, |V_tgt|)
        # |output| = (bs, m, |V_tgt|)
        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            # Based on the length information, gererate mask to prevent that
            # shorter sample has wasted attention.
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length) - source와 같음
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # Get word embedding vectors for every time-step of input sentence.
        emb_src = self.emb_src(x) # 원핫벡터를 넣으면 embedding 벡터가 나오는 것.
        # |emb_src| = (batch_size, length, word_vec_size)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size) # 인코더의 전체 타임스탭 마지막 레이어의 히든 스테이트
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2) # 인코더의 마지막 타임스텝의 전체 레이어의 히든 스테이트

        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        emb_tgt = self.emb_dec(tgt) # teacher forcing이기 때문에 정답을 한꺼번에 embedding vector로 만들어 준다.
        # |emb_tgt| = (batch_size, length, word_vec_size)
        h_tilde = []

        h_t_tilde = None # 첫 번째 타임스탭은 이전 타임 스탭의 틸다 값이 없다.
        decoder_hidden = h_0_tgt
        # Run decoder until the end of the time-step.
        for t in range(tgt.size(1)):
            # Teacher Forcing: take each input from training set,
            # not from the last time-step's output.
            # Because of Teacher Forcing,
            # training procedure and inference procedure becomes different.
            # Of course, because of sequential running in decoder,
            # this causes severe bottle-neck.
            emb_t = emb_tgt[:, t, :].unsqueeze(1) # 가운데 1이 날라가기 때문에 unsqueeze(1)로 다시 채워주는 것
            # |emb_t| = (batch_size, 1, word_vec_size)
            # |h_t_tilde| = (batch_size, 1, hidden_size) # 이전 스탭 틸다임.

            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden
                                                          )
            # |decoder_output| = (batch_size, 1, hidden_size)
            # |decoder_hidden| = (n_layers, batch_size, hidden_size)

            context_vector = self.attn(h_src, decoder_output, mask) #h_src는 인코더의 전체 타임스탭의 아웃풋
            # |context_vector| = (batch_size, 1, hidden_size)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector
                                                         ], dim=-1)))
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1)
        # |h_tilde| = (batch_size, length, hidden_size)

        y_hat = self.generator(h_tilde)
        # |y_hat| = (batch_size, length, output_size)

        return y_hat
