  .text
  .globl Variable_Create
  .type Variable_Create, @function

#! file-offset 0x68de0
#! rip-offset  0x28de0
#! capacity    320 bytes

# Text                          #  Line  RIP      Bytes  Opcode              
.Variable_Create:               #        0x28de0  0      OPC=<label>         
  pushq %r13                    #  1     0x28de0  2      OPC=pushq_r64_1     
  pushq %r12                    #  2     0x28de2  2      OPC=pushq_r64_1     
  pushq %rbx                    #  3     0x28de4  1      OPC=pushq_r64_1     
  movl %esi, %r13d              #  4     0x28de5  3      OPC=movl_r32_r32    
  movl %edi, %r12d              #  5     0x28de8  3      OPC=movl_r32_r32    
  movl $0x24, %edi              #  6     0x28deb  5      OPC=movl_r32_imm32  
  nop                           #  7     0x28df0  1      OPC=nop             
  nop                           #  8     0x28df1  1      OPC=nop             
  nop                           #  9     0x28df2  1      OPC=nop             
  nop                           #  10    0x28df3  1      OPC=nop             
  nop                           #  11    0x28df4  1      OPC=nop             
  nop                           #  12    0x28df5  1      OPC=nop             
  nop                           #  13    0x28df6  1      OPC=nop             
  nop                           #  14    0x28df7  1      OPC=nop             
  nop                           #  15    0x28df8  1      OPC=nop             
  nop                           #  16    0x28df9  1      OPC=nop             
  nop                           #  17    0x28dfa  1      OPC=nop             
  callq .malloc                 #  18    0x28dfb  5      OPC=callq_label     
  movl %eax, %ebx               #  19    0x28e00  2      OPC=movl_r32_r32    
  testq %rbx, %rbx              #  20    0x28e02  3      OPC=testq_r64_r64   
  je .L_28ee0                   #  21    0x28e05  6      OPC=je_label_1      
  nop                           #  22    0x28e0b  1      OPC=nop             
  nop                           #  23    0x28e0c  1      OPC=nop             
  nop                           #  24    0x28e0d  1      OPC=nop             
  nop                           #  25    0x28e0e  1      OPC=nop             
  nop                           #  26    0x28e0f  1      OPC=nop             
  nop                           #  27    0x28e10  1      OPC=nop             
  nop                           #  28    0x28e11  1      OPC=nop             
  nop                           #  29    0x28e12  1      OPC=nop             
  nop                           #  30    0x28e13  1      OPC=nop             
  nop                           #  31    0x28e14  1      OPC=nop             
  nop                           #  32    0x28e15  1      OPC=nop             
  nop                           #  33    0x28e16  1      OPC=nop             
  nop                           #  34    0x28e17  1      OPC=nop             
  nop                           #  35    0x28e18  1      OPC=nop             
  nop                           #  36    0x28e19  1      OPC=nop             
  nop                           #  37    0x28e1a  1      OPC=nop             
  nop                           #  38    0x28e1b  1      OPC=nop             
  nop                           #  39    0x28e1c  1      OPC=nop             
  nop                           #  40    0x28e1d  1      OPC=nop             
  nop                           #  41    0x28e1e  1      OPC=nop             
  nop                           #  42    0x28e1f  1      OPC=nop             
.L_28e20:                       #        0x28e20  0      OPC=<label>         
  movl %ebx, %ebx               #  43    0x28e20  2      OPC=movl_r32_r32    
  movl %r13d, (%r15,%rbx,1)     #  44    0x28e22  4      OPC=movl_m32_r32    
  movl $0x2, %edi               #  45    0x28e26  5      OPC=movl_r32_imm32  
  nop                           #  46    0x28e2b  1      OPC=nop             
  nop                           #  47    0x28e2c  1      OPC=nop             
  nop                           #  48    0x28e2d  1      OPC=nop             
  nop                           #  49    0x28e2e  1      OPC=nop             
  nop                           #  50    0x28e2f  1      OPC=nop             
  nop                           #  51    0x28e30  1      OPC=nop             
  nop                           #  52    0x28e31  1      OPC=nop             
  nop                           #  53    0x28e32  1      OPC=nop             
  nop                           #  54    0x28e33  1      OPC=nop             
  nop                           #  55    0x28e34  1      OPC=nop             
  nop                           #  56    0x28e35  1      OPC=nop             
  nop                           #  57    0x28e36  1      OPC=nop             
  nop                           #  58    0x28e37  1      OPC=nop             
  nop                           #  59    0x28e38  1      OPC=nop             
  nop                           #  60    0x28e39  1      OPC=nop             
  nop                           #  61    0x28e3a  1      OPC=nop             
  callq .List_Create            #  62    0x28e3b  5      OPC=callq_label     
  movl %ebx, %ebx               #  63    0x28e40  2      OPC=movl_r32_r32    
  movl %eax, 0x4(%r15,%rbx,1)   #  64    0x28e42  5      OPC=movl_m32_r32    
  movl %ebx, %ebx               #  65    0x28e47  2      OPC=movl_r32_r32    
  movl $0x0, 0x8(%r15,%rbx,1)   #  66    0x28e49  9      OPC=movl_m32_imm32  
  movl %ebx, %ebx               #  67    0x28e52  2      OPC=movl_r32_r32    
  movl $0x0, 0xc(%r15,%rbx,1)   #  68    0x28e54  9      OPC=movl_m32_imm32  
  nop                           #  69    0x28e5d  1      OPC=nop             
  nop                           #  70    0x28e5e  1      OPC=nop             
  nop                           #  71    0x28e5f  1      OPC=nop             
  movl %ebx, %ebx               #  72    0x28e60  2      OPC=movl_r32_r32    
  movl $0x6, 0x10(%r15,%rbx,1)  #  73    0x28e62  9      OPC=movl_m32_imm32  
  movl %ebx, %ebx               #  74    0x28e6b  2      OPC=movl_r32_r32    
  movl $0x1, 0x14(%r15,%rbx,1)  #  75    0x28e6d  9      OPC=movl_m32_imm32  
  leal 0x18(%rbx), %edi         #  76    0x28e76  3      OPC=leal_r32_m16    
  movl $0xa, %edx               #  77    0x28e79  5      OPC=movl_r32_imm32  
  xchgw %ax, %ax                #  78    0x28e7e  2      OPC=xchgw_ax_r16    
  movl %r12d, %esi              #  79    0x28e80  3      OPC=movl_r32_r32    
  nop                           #  80    0x28e83  1      OPC=nop             
  nop                           #  81    0x28e84  1      OPC=nop             
  nop                           #  82    0x28e85  1      OPC=nop             
  nop                           #  83    0x28e86  1      OPC=nop             
  nop                           #  84    0x28e87  1      OPC=nop             
  nop                           #  85    0x28e88  1      OPC=nop             
  nop                           #  86    0x28e89  1      OPC=nop             
  nop                           #  87    0x28e8a  1      OPC=nop             
  nop                           #  88    0x28e8b  1      OPC=nop             
  nop                           #  89    0x28e8c  1      OPC=nop             
  nop                           #  90    0x28e8d  1      OPC=nop             
  nop                           #  91    0x28e8e  1      OPC=nop             
  nop                           #  92    0x28e8f  1      OPC=nop             
  nop                           #  93    0x28e90  1      OPC=nop             
  nop                           #  94    0x28e91  1      OPC=nop             
  nop                           #  95    0x28e92  1      OPC=nop             
  nop                           #  96    0x28e93  1      OPC=nop             
  nop                           #  97    0x28e94  1      OPC=nop             
  nop                           #  98    0x28e95  1      OPC=nop             
  nop                           #  99    0x28e96  1      OPC=nop             
  nop                           #  100   0x28e97  1      OPC=nop             
  nop                           #  101   0x28e98  1      OPC=nop             
  nop                           #  102   0x28e99  1      OPC=nop             
  nop                           #  103   0x28e9a  1      OPC=nop             
  callq .strncpy                #  104   0x28e9b  5      OPC=callq_label     
  movl %ebx, %ebx               #  105   0x28ea0  2      OPC=movl_r32_r32    
  movb $0x0, 0x21(%r15,%rbx,1)  #  106   0x28ea2  6      OPC=movb_m8_imm8    
  movl %ebx, %edi               #  107   0x28ea8  2      OPC=movl_r32_r32    
  xchgw %ax, %ax                #  108   0x28eaa  2      OPC=xchgw_ax_r16    
  nop                           #  109   0x28eac  1      OPC=nop             
  nop                           #  110   0x28ead  1      OPC=nop             
  nop                           #  111   0x28eae  1      OPC=nop             
  nop                           #  112   0x28eaf  1      OPC=nop             
  nop                           #  113   0x28eb0  1      OPC=nop             
  nop                           #  114   0x28eb1  1      OPC=nop             
  nop                           #  115   0x28eb2  1      OPC=nop             
  nop                           #  116   0x28eb3  1      OPC=nop             
  nop                           #  117   0x28eb4  1      OPC=nop             
  nop                           #  118   0x28eb5  1      OPC=nop             
  nop                           #  119   0x28eb6  1      OPC=nop             
  nop                           #  120   0x28eb7  1      OPC=nop             
  nop                           #  121   0x28eb8  1      OPC=nop             
  nop                           #  122   0x28eb9  1      OPC=nop             
  nop                           #  123   0x28eba  1      OPC=nop             
  callq .AddVariable            #  124   0x28ebb  5      OPC=callq_label     
  movl %ebx, %eax               #  125   0x28ec0  2      OPC=movl_r32_r32    
  popq %rbx                     #  126   0x28ec2  1      OPC=popq_r64_1      
  popq %r12                     #  127   0x28ec3  2      OPC=popq_r64_1      
  popq %r13                     #  128   0x28ec5  2      OPC=popq_r64_1      
  popq %r11                     #  129   0x28ec7  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d       #  130   0x28ec9  7      OPC=andl_r32_imm32  
  nop                           #  131   0x28ed0  1      OPC=nop             
  nop                           #  132   0x28ed1  1      OPC=nop             
  nop                           #  133   0x28ed2  1      OPC=nop             
  nop                           #  134   0x28ed3  1      OPC=nop             
  addq %r15, %r11               #  135   0x28ed4  3      OPC=addq_r64_r64    
  jmpq %r11                     #  136   0x28ed7  3      OPC=jmpq_r64        
  nop                           #  137   0x28eda  1      OPC=nop             
  nop                           #  138   0x28edb  1      OPC=nop             
  nop                           #  139   0x28edc  1      OPC=nop             
  nop                           #  140   0x28edd  1      OPC=nop             
  nop                           #  141   0x28ede  1      OPC=nop             
  nop                           #  142   0x28edf  1      OPC=nop             
  nop                           #  143   0x28ee0  1      OPC=nop             
  nop                           #  144   0x28ee1  1      OPC=nop             
  nop                           #  145   0x28ee2  1      OPC=nop             
  nop                           #  146   0x28ee3  1      OPC=nop             
  nop                           #  147   0x28ee4  1      OPC=nop             
  nop                           #  148   0x28ee5  1      OPC=nop             
  nop                           #  149   0x28ee6  1      OPC=nop             
.L_28ee0:                       #        0x28ee7  0      OPC=<label>         
  movl $0x10020ae1, %edi        #  150   0x28ee7  5      OPC=movl_r32_imm32  
  nop                           #  151   0x28eec  1      OPC=nop             
  nop                           #  152   0x28eed  1      OPC=nop             
  nop                           #  153   0x28eee  1      OPC=nop             
  nop                           #  154   0x28eef  1      OPC=nop             
  nop                           #  155   0x28ef0  1      OPC=nop             
  nop                           #  156   0x28ef1  1      OPC=nop             
  nop                           #  157   0x28ef2  1      OPC=nop             
  nop                           #  158   0x28ef3  1      OPC=nop             
  nop                           #  159   0x28ef4  1      OPC=nop             
  nop                           #  160   0x28ef5  1      OPC=nop             
  nop                           #  161   0x28ef6  1      OPC=nop             
  nop                           #  162   0x28ef7  1      OPC=nop             
  nop                           #  163   0x28ef8  1      OPC=nop             
  nop                           #  164   0x28ef9  1      OPC=nop             
  nop                           #  165   0x28efa  1      OPC=nop             
  nop                           #  166   0x28efb  1      OPC=nop             
  nop                           #  167   0x28efc  1      OPC=nop             
  nop                           #  168   0x28efd  1      OPC=nop             
  nop                           #  169   0x28efe  1      OPC=nop             
  nop                           #  170   0x28eff  1      OPC=nop             
  nop                           #  171   0x28f00  1      OPC=nop             
  nop                           #  172   0x28f01  1      OPC=nop             
  callq .Error                  #  173   0x28f02  5      OPC=callq_label     
  jmpq .L_28e20                 #  174   0x28f07  5      OPC=jmpq_label_1    
  nop                           #  175   0x28f0c  1      OPC=nop             
  nop                           #  176   0x28f0d  1      OPC=nop             
  nop                           #  177   0x28f0e  1      OPC=nop             
  nop                           #  178   0x28f0f  1      OPC=nop             
  nop                           #  179   0x28f10  1      OPC=nop             
  nop                           #  180   0x28f11  1      OPC=nop             
  nop                           #  181   0x28f12  1      OPC=nop             
  nop                           #  182   0x28f13  1      OPC=nop             
  nop                           #  183   0x28f14  1      OPC=nop             
  nop                           #  184   0x28f15  1      OPC=nop             
  nop                           #  185   0x28f16  1      OPC=nop             
  nop                           #  186   0x28f17  1      OPC=nop             
  nop                           #  187   0x28f18  1      OPC=nop             
  nop                           #  188   0x28f19  1      OPC=nop             
  nop                           #  189   0x28f1a  1      OPC=nop             
  nop                           #  190   0x28f1b  1      OPC=nop             
  nop                           #  191   0x28f1c  1      OPC=nop             
  nop                           #  192   0x28f1d  1      OPC=nop             
  nop                           #  193   0x28f1e  1      OPC=nop             
  nop                           #  194   0x28f1f  1      OPC=nop             
  nop                           #  195   0x28f20  1      OPC=nop             
  nop                           #  196   0x28f21  1      OPC=nop             
  nop                           #  197   0x28f22  1      OPC=nop             
  nop                           #  198   0x28f23  1      OPC=nop             
  nop                           #  199   0x28f24  1      OPC=nop             
  nop                           #  200   0x28f25  1      OPC=nop             
  nop                           #  201   0x28f26  1      OPC=nop             
                                                                             
.size Variable_Create, .-Variable_Create

