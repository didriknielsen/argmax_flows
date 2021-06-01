from .ar.flow import ArgmaxARFlow
from .coupling.flow import ArgmaxCouplingFlow


def add_model_args(parser):

    parser.add_argument('--model', type=str, default='ar', choices={'ar', 'coupling'})
    tmp_args, _ = parser.parse_known_args()
    if tmp_args.model == 'ar':
        # Flow params
        parser.add_argument('--num_steps', type=int, default=4)
        parser.add_argument('--actnorm', type=eval, default=False)
        parser.add_argument('--perm_channel', type=str, default='none', choices={'conv', 'shuffle', 'none'})
        parser.add_argument('--perm_length', type=str, default='reverse', choices={'reverse', 'none'})
        parser.add_argument('--base_dist', type=str, default='conv_gauss', choices={'conv_gauss', 'gauss', 'gumbel'})

        # Encoder params
        parser.add_argument('--encoder_steps', type=int, default=2)
        parser.add_argument('--encoder_bins', type=int, default=5)

        # Context params
        parser.add_argument('--context_size', type=int, default=128)
        parser.add_argument('--context_lstm_layers', type=int, default=1)
        parser.add_argument('--context_lstm_size', type=int, default=512)

        # Network params
        parser.add_argument('--lstm_layers', type=int, default=1)
        parser.add_argument('--lstm_size', type=int, default=512)
        parser.add_argument('--lstm_dropout', type=float, default=0.0)
        parser.add_argument('--input_dp_rate', type=float, default=0.0)

    elif tmp_args.model == 'coupling':
        # Flow params
        parser.add_argument('--num_steps', type=int, default=1)
        parser.add_argument('--actnorm', type=eval, default=False)
        parser.add_argument('--num_mixtures', type=int, default=27)
        parser.add_argument('--perm_channel', type=str, default='conv', choices={'conv', 'shuffle', 'none'})
        parser.add_argument('--perm_length', type=str, default='reverse', choices={'reverse', 'none'})
        parser.add_argument('--base_dist', type=str, default='conv_gauss', choices={'conv_gauss', 'gauss', 'gumbel'})

        # Encoder params
        parser.add_argument('--encoder_steps', type=int, default=2)
        parser.add_argument('--encoder_bins', type=int, default=None)
        parser.add_argument('--encoder_ff_size', type=int, default=1024)

        # Context params
        parser.add_argument('--context_size', type=int, default=128)
        parser.add_argument('--context_ff_layers', type=int, default=1)
        parser.add_argument('--context_ff_size', type=int, default=512)
        parser.add_argument('--context_dropout', type=float, default=0.0)

        # Network params
        parser.add_argument('--lstm_layers', type=int, default=1)
        parser.add_argument('--lstm_size', type=int, default=512)
        parser.add_argument('--lstm_dropout', type=float, default=0.0)
        parser.add_argument('--input_dp_rate', type=float, default=0.0)



def get_model_id(args):
    if args.model == 'ar':
        return'argmax_ar'
    elif args.model == 'coupling':
        return 'argmax_coupling'


def get_model(args, data_shape, num_classes):

    if args.model == 'ar':
        return ArgmaxARFlow(
            data_shape=data_shape,
            num_classes=num_classes,
            num_steps=args.num_steps,
            actnorm=args.actnorm,
            perm_channel=args.perm_channel,
            perm_length=args.perm_length,
            base_dist=args.base_dist,
            encoder_steps=args.encoder_steps,
            encoder_bins=args.encoder_bins,
            context_size=args.context_size,
            lstm_layers=args.lstm_layers,
            lstm_size=args.lstm_size,
            lstm_dropout=args.lstm_dropout,
            context_lstm_layers=args.context_lstm_layers,
            context_lstm_size=args.context_lstm_size,
            input_dp_rate=args.input_dp_rate
        )

    elif args.model == 'coupling':
        return ArgmaxCouplingFlow(
            data_shape=data_shape,
            num_classes=num_classes,
            num_steps=args.num_steps,
            actnorm=args.actnorm,
            num_mixtures=args.num_mixtures,
            perm_channel=args.perm_channel,
            perm_length=args.perm_length,
            base_dist=args.base_dist,
            encoder_steps=args.encoder_steps,
            encoder_bins=args.encoder_bins,
            encoder_ff_size=args.encoder_ff_size,
            context_size=args.context_size,
            context_ff_layers=args.context_ff_layers,
            context_ff_size=args.context_ff_size,
            context_dropout=args.context_dropout,
            lstm_layers=args.lstm_layers,
            lstm_size=args.lstm_size,
            lstm_dropout=args.lstm_dropout,
            input_dp_rate=args.input_dp_rate
        )
