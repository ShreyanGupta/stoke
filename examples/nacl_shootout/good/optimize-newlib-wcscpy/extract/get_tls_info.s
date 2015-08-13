  .text
  .globl get_tls_info
  .type get_tls_info, @function

#! file-offset 0x158980
#! rip-offset  0x118980
#! capacity    672 bytes

# Text                          #  Line  RIP       Bytes  Opcode              
.get_tls_info:                  #        0x118980  0      OPC=<label>         
  movl 0xff603e6(%rip), %eax    #  1     0x118980  6      OPC=movl_r32_m32    
  testl %eax, %eax              #  2     0x118986  2      OPC=testl_r32_r32   
  jne .L_118a00                 #  3     0x118988  2      OPC=jne_label       
  movl $0x10020000, %eax        #  4     0x11898a  5      OPC=movl_r32_imm32  
  testl %eax, %eax              #  5     0x11898f  2      OPC=testl_r32_r32   
  jne .L_118a20                 #  6     0x118991  6      OPC=jne_label_1     
  nop                           #  7     0x118997  1      OPC=nop             
  nop                           #  8     0x118998  1      OPC=nop             
  nop                           #  9     0x118999  1      OPC=nop             
  nop                           #  10    0x11899a  1      OPC=nop             
  nop                           #  11    0x11899b  1      OPC=nop             
  nop                           #  12    0x11899c  1      OPC=nop             
  nop                           #  13    0x11899d  1      OPC=nop             
  nop                           #  14    0x11899e  1      OPC=nop             
  nop                           #  15    0x11899f  1      OPC=nop             
.L_1189a0:                      #        0x1189a0  0      OPC=<label>         
  testl %eax, %eax              #  16    0x1189a0  2      OPC=testl_r32_r32   
  jne .L_118b20                 #  17    0x1189a2  6      OPC=jne_label_1     
  nop                           #  18    0x1189a8  1      OPC=nop             
  nop                           #  19    0x1189a9  1      OPC=nop             
  nop                           #  20    0x1189aa  1      OPC=nop             
  nop                           #  21    0x1189ab  1      OPC=nop             
  nop                           #  22    0x1189ac  1      OPC=nop             
  nop                           #  23    0x1189ad  1      OPC=nop             
  nop                           #  24    0x1189ae  1      OPC=nop             
  nop                           #  25    0x1189af  1      OPC=nop             
  nop                           #  26    0x1189b0  1      OPC=nop             
  nop                           #  27    0x1189b1  1      OPC=nop             
  nop                           #  28    0x1189b2  1      OPC=nop             
  nop                           #  29    0x1189b3  1      OPC=nop             
  nop                           #  30    0x1189b4  1      OPC=nop             
  nop                           #  31    0x1189b5  1      OPC=nop             
  nop                           #  32    0x1189b6  1      OPC=nop             
  nop                           #  33    0x1189b7  1      OPC=nop             
  nop                           #  34    0x1189b8  1      OPC=nop             
  nop                           #  35    0x1189b9  1      OPC=nop             
  nop                           #  36    0x1189ba  1      OPC=nop             
  nop                           #  37    0x1189bb  1      OPC=nop             
  nop                           #  38    0x1189bc  1      OPC=nop             
  nop                           #  39    0x1189bd  1      OPC=nop             
  nop                           #  40    0x1189be  1      OPC=nop             
  nop                           #  41    0x1189bf  1      OPC=nop             
.L_1189c0:                      #        0x1189c0  0      OPC=<label>         
  movl -0x1189c6(%rip), %eax    #  42    0x1189c0  6      OPC=movl_r32_m32    
  movl $0x0, 0xff60390(%rip)    #  43    0x1189c6  10     OPC=movl_m32_imm32  
  movl %eax, 0xff60396(%rip)    #  44    0x1189d0  6      OPC=movl_m32_r32    
  movl $0x0, %eax               #  45    0x1189d6  5      OPC=movl_r32_imm32  
  subl $0x0, %eax               #  46    0x1189db  5      OPC=subl_eax_imm32  
  movl %eax, 0xff6037e(%rip)    #  47    0x1189e0  6      OPC=movl_m32_r32    
  movl $0x0, %eax               #  48    0x1189e6  5      OPC=movl_r32_imm32  
  subl $0x0, %eax               #  49    0x1189eb  5      OPC=subl_eax_imm32  
  movl %eax, 0xff60372(%rip)    #  50    0x1189f0  6      OPC=movl_m32_r32    
  nop                           #  51    0x1189f6  1      OPC=nop             
  nop                           #  52    0x1189f7  1      OPC=nop             
  nop                           #  53    0x1189f8  1      OPC=nop             
  nop                           #  54    0x1189f9  1      OPC=nop             
  nop                           #  55    0x1189fa  1      OPC=nop             
  nop                           #  56    0x1189fb  1      OPC=nop             
  nop                           #  57    0x1189fc  1      OPC=nop             
  nop                           #  58    0x1189fd  1      OPC=nop             
  nop                           #  59    0x1189fe  1      OPC=nop             
  nop                           #  60    0x1189ff  1      OPC=nop             
.L_118a00:                      #        0x118a00  0      OPC=<label>         
  popq %r11                     #  61    0x118a00  2      OPC=popq_r64_1      
  movl $0x10078d60, %eax        #  62    0x118a02  5      OPC=movl_r32_imm32  
  andl $0xffffffe0, %r11d       #  63    0x118a07  7      OPC=andl_r32_imm32  
  nop                           #  64    0x118a0e  1      OPC=nop             
  nop                           #  65    0x118a0f  1      OPC=nop             
  nop                           #  66    0x118a10  1      OPC=nop             
  nop                           #  67    0x118a11  1      OPC=nop             
  addq %r15, %r11               #  68    0x118a12  3      OPC=addq_r64_r64    
  jmpq %r11                     #  69    0x118a15  3      OPC=jmpq_r64        
  nop                           #  70    0x118a18  1      OPC=nop             
  nop                           #  71    0x118a19  1      OPC=nop             
  nop                           #  72    0x118a1a  1      OPC=nop             
  nop                           #  73    0x118a1b  1      OPC=nop             
  nop                           #  74    0x118a1c  1      OPC=nop             
  nop                           #  75    0x118a1d  1      OPC=nop             
  nop                           #  76    0x118a1e  1      OPC=nop             
  nop                           #  77    0x118a1f  1      OPC=nop             
  nop                           #  78    0x118a20  1      OPC=nop             
  nop                           #  79    0x118a21  1      OPC=nop             
  nop                           #  80    0x118a22  1      OPC=nop             
  nop                           #  81    0x118a23  1      OPC=nop             
  nop                           #  82    0x118a24  1      OPC=nop             
  nop                           #  83    0x118a25  1      OPC=nop             
  nop                           #  84    0x118a26  1      OPC=nop             
.L_118a20:                      #        0x118a27  0      OPC=<label>         
  cmpb $0x1, 0xff075dd(%rip)    #  85    0x118a27  7      OPC=cmpb_m8_imm8    
  jne .L_1189a0                 #  86    0x118a2e  6      OPC=jne_label_1     
  cmpw $0x20, 0xff075f5(%rip)   #  87    0x118a34  8      OPC=cmpw_m16_imm8   
  jne .L_1189a0                 #  88    0x118a3c  6      OPC=jne_label_1     
  nop                           #  89    0x118a42  1      OPC=nop             
  nop                           #  90    0x118a43  1      OPC=nop             
  nop                           #  91    0x118a44  1      OPC=nop             
  nop                           #  92    0x118a45  1      OPC=nop             
  nop                           #  93    0x118a46  1      OPC=nop             
  movl 0xff075d6(%rip), %esi    #  94    0x118a47  6      OPC=movl_r32_m32    
  movzwl 0xff075df(%rip), %edi  #  95    0x118a4d  7      OPC=movzwl_r32_m16  
  xorl %ecx, %ecx               #  96    0x118a54  2      OPC=xorl_r32_r32    
  addl %eax, %esi               #  97    0x118a56  2      OPC=addl_r32_r32    
  jmpq .L_118a80                #  98    0x118a58  2      OPC=jmpq_label      
  nop                           #  99    0x118a5a  1      OPC=nop             
  nop                           #  100   0x118a5b  1      OPC=nop             
  nop                           #  101   0x118a5c  1      OPC=nop             
  nop                           #  102   0x118a5d  1      OPC=nop             
  nop                           #  103   0x118a5e  1      OPC=nop             
  nop                           #  104   0x118a5f  1      OPC=nop             
  nop                           #  105   0x118a60  1      OPC=nop             
  nop                           #  106   0x118a61  1      OPC=nop             
  nop                           #  107   0x118a62  1      OPC=nop             
  nop                           #  108   0x118a63  1      OPC=nop             
  nop                           #  109   0x118a64  1      OPC=nop             
  nop                           #  110   0x118a65  1      OPC=nop             
  nop                           #  111   0x118a66  1      OPC=nop             
.L_118a60:                      #        0x118a67  0      OPC=<label>         
  addl $0x1, %ecx               #  112   0x118a67  3      OPC=addl_r32_imm8   
  nop                           #  113   0x118a6a  1      OPC=nop             
  nop                           #  114   0x118a6b  1      OPC=nop             
  nop                           #  115   0x118a6c  1      OPC=nop             
  nop                           #  116   0x118a6d  1      OPC=nop             
  nop                           #  117   0x118a6e  1      OPC=nop             
  nop                           #  118   0x118a6f  1      OPC=nop             
  nop                           #  119   0x118a70  1      OPC=nop             
  nop                           #  120   0x118a71  1      OPC=nop             
  nop                           #  121   0x118a72  1      OPC=nop             
  nop                           #  122   0x118a73  1      OPC=nop             
  nop                           #  123   0x118a74  1      OPC=nop             
  nop                           #  124   0x118a75  1      OPC=nop             
  nop                           #  125   0x118a76  1      OPC=nop             
  nop                           #  126   0x118a77  1      OPC=nop             
  nop                           #  127   0x118a78  1      OPC=nop             
  nop                           #  128   0x118a79  1      OPC=nop             
  nop                           #  129   0x118a7a  1      OPC=nop             
  nop                           #  130   0x118a7b  1      OPC=nop             
  nop                           #  131   0x118a7c  1      OPC=nop             
  nop                           #  132   0x118a7d  1      OPC=nop             
  nop                           #  133   0x118a7e  1      OPC=nop             
  nop                           #  134   0x118a7f  1      OPC=nop             
  nop                           #  135   0x118a80  1      OPC=nop             
  nop                           #  136   0x118a81  1      OPC=nop             
  nop                           #  137   0x118a82  1      OPC=nop             
  nop                           #  138   0x118a83  1      OPC=nop             
  nop                           #  139   0x118a84  1      OPC=nop             
  nop                           #  140   0x118a85  1      OPC=nop             
  nop                           #  141   0x118a86  1      OPC=nop             
.L_118a80:                      #        0x118a87  0      OPC=<label>         
  cmpl %edi, %ecx               #  142   0x118a87  2      OPC=cmpl_r32_r32    
  jge .L_1189a0                 #  143   0x118a89  6      OPC=jge_label_1     
  movl %esi, %edx               #  144   0x118a8f  2      OPC=movl_r32_r32    
  addl $0x20, %esi              #  145   0x118a91  3      OPC=addl_r32_imm8   
  movl %edx, %edx               #  146   0x118a94  2      OPC=movl_r32_r32    
  cmpl $0x7, (%r15,%rdx,1)      #  147   0x118a96  5      OPC=cmpl_m32_imm8   
  jne .L_118a60                 #  148   0x118a9b  2      OPC=jne_label       
  movl %edx, %edx               #  149   0x118a9d  2      OPC=movl_r32_r32    
  movl 0x1c(%r15,%rdx,1), %eax  #  150   0x118a9f  5      OPC=movl_r32_m32    
  nop                           #  151   0x118aa4  1      OPC=nop             
  nop                           #  152   0x118aa5  1      OPC=nop             
  nop                           #  153   0x118aa6  1      OPC=nop             
  cmpw $0x3, 0xff07568(%rip)    #  154   0x118aa7  8      OPC=cmpw_m16_imm8   
  movl %eax, 0xff602be(%rip)    #  155   0x118aaf  6      OPC=movl_m32_r32    
  movl %edx, %edx               #  156   0x118ab5  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rdx,1), %eax   #  157   0x118ab7  5      OPC=movl_r32_m32    
  movl %eax, 0xff602a5(%rip)    #  158   0x118abc  6      OPC=movl_m32_r32    
  nop                           #  159   0x118ac2  1      OPC=nop             
  nop                           #  160   0x118ac3  1      OPC=nop             
  nop                           #  161   0x118ac4  1      OPC=nop             
  nop                           #  162   0x118ac5  1      OPC=nop             
  nop                           #  163   0x118ac6  1      OPC=nop             
  jne .L_118ae0                 #  164   0x118ac7  2      OPC=jne_label       
  addl $0x10020000, %eax        #  165   0x118ac9  5      OPC=addl_eax_imm32  
  movl %eax, 0xff60293(%rip)    #  166   0x118ace  6      OPC=movl_m32_r32    
  nop                           #  167   0x118ad4  1      OPC=nop             
  nop                           #  168   0x118ad5  1      OPC=nop             
  nop                           #  169   0x118ad6  1      OPC=nop             
  nop                           #  170   0x118ad7  1      OPC=nop             
  nop                           #  171   0x118ad8  1      OPC=nop             
  nop                           #  172   0x118ad9  1      OPC=nop             
  nop                           #  173   0x118ada  1      OPC=nop             
  nop                           #  174   0x118adb  1      OPC=nop             
  nop                           #  175   0x118adc  1      OPC=nop             
  nop                           #  176   0x118add  1      OPC=nop             
  nop                           #  177   0x118ade  1      OPC=nop             
  nop                           #  178   0x118adf  1      OPC=nop             
  nop                           #  179   0x118ae0  1      OPC=nop             
  nop                           #  180   0x118ae1  1      OPC=nop             
  nop                           #  181   0x118ae2  1      OPC=nop             
  nop                           #  182   0x118ae3  1      OPC=nop             
  nop                           #  183   0x118ae4  1      OPC=nop             
  nop                           #  184   0x118ae5  1      OPC=nop             
  nop                           #  185   0x118ae6  1      OPC=nop             
.L_118ae0:                      #        0x118ae7  0      OPC=<label>         
  movl %edx, %edx               #  186   0x118ae7  2      OPC=movl_r32_r32    
  movl 0x10(%r15,%rdx,1), %ecx  #  187   0x118ae9  5      OPC=movl_r32_m32    
  movl %ecx, 0xff60277(%rip)    #  188   0x118aee  6      OPC=movl_m32_r32    
  movl %edx, %edx               #  189   0x118af4  2      OPC=movl_r32_r32    
  movl 0x14(%r15,%rdx,1), %eax  #  190   0x118af6  5      OPC=movl_r32_m32    
  subl %ecx, %eax               #  191   0x118afb  2      OPC=subl_r32_r32    
  movl %eax, 0xff6026c(%rip)    #  192   0x118afd  6      OPC=movl_m32_r32    
  nop                           #  193   0x118b03  1      OPC=nop             
  nop                           #  194   0x118b04  1      OPC=nop             
  nop                           #  195   0x118b05  1      OPC=nop             
  nop                           #  196   0x118b06  1      OPC=nop             
  jmpq .L_118a00                #  197   0x118b07  5      OPC=jmpq_label_1    
  nop                           #  198   0x118b0c  1      OPC=nop             
  nop                           #  199   0x118b0d  1      OPC=nop             
  nop                           #  200   0x118b0e  1      OPC=nop             
  nop                           #  201   0x118b0f  1      OPC=nop             
  nop                           #  202   0x118b10  1      OPC=nop             
  nop                           #  203   0x118b11  1      OPC=nop             
  nop                           #  204   0x118b12  1      OPC=nop             
  nop                           #  205   0x118b13  1      OPC=nop             
  nop                           #  206   0x118b14  1      OPC=nop             
  nop                           #  207   0x118b15  1      OPC=nop             
  nop                           #  208   0x118b16  1      OPC=nop             
  nop                           #  209   0x118b17  1      OPC=nop             
  nop                           #  210   0x118b18  1      OPC=nop             
  nop                           #  211   0x118b19  1      OPC=nop             
  nop                           #  212   0x118b1a  1      OPC=nop             
  nop                           #  213   0x118b1b  1      OPC=nop             
  nop                           #  214   0x118b1c  1      OPC=nop             
  nop                           #  215   0x118b1d  1      OPC=nop             
  nop                           #  216   0x118b1e  1      OPC=nop             
  nop                           #  217   0x118b1f  1      OPC=nop             
  nop                           #  218   0x118b20  1      OPC=nop             
  nop                           #  219   0x118b21  1      OPC=nop             
  nop                           #  220   0x118b22  1      OPC=nop             
  nop                           #  221   0x118b23  1      OPC=nop             
  nop                           #  222   0x118b24  1      OPC=nop             
  nop                           #  223   0x118b25  1      OPC=nop             
  nop                           #  224   0x118b26  1      OPC=nop             
.L_118b20:                      #        0x118b27  0      OPC=<label>         
  cmpb $0x2, 0xff074dd(%rip)    #  225   0x118b27  7      OPC=cmpb_m8_imm8    
  jne .L_1189c0                 #  226   0x118b2e  6      OPC=jne_label_1     
  cmpw $0x38, 0xff07501(%rip)   #  227   0x118b34  8      OPC=cmpw_m16_imm8   
  jne .L_1189c0                 #  228   0x118b3c  6      OPC=jne_label_1     
  nop                           #  229   0x118b42  1      OPC=nop             
  nop                           #  230   0x118b43  1      OPC=nop             
  nop                           #  231   0x118b44  1      OPC=nop             
  nop                           #  232   0x118b45  1      OPC=nop             
  nop                           #  233   0x118b46  1      OPC=nop             
  movq 0xff074d9(%rip), %rcx    #  234   0x118b47  7      OPC=movq_r64_m64    
  movzwl 0xff074ea(%rip), %esi  #  235   0x118b4e  7      OPC=movzwl_r32_m16  
  xorl %edx, %edx               #  236   0x118b55  2      OPC=xorl_r32_r32    
  addl $0x10020000, %ecx        #  237   0x118b57  6      OPC=addl_r32_imm32  
  jmpq .L_118b80                #  238   0x118b5d  2      OPC=jmpq_label      
  nop                           #  239   0x118b5f  1      OPC=nop             
  nop                           #  240   0x118b60  1      OPC=nop             
  nop                           #  241   0x118b61  1      OPC=nop             
  nop                           #  242   0x118b62  1      OPC=nop             
  nop                           #  243   0x118b63  1      OPC=nop             
  nop                           #  244   0x118b64  1      OPC=nop             
  nop                           #  245   0x118b65  1      OPC=nop             
  nop                           #  246   0x118b66  1      OPC=nop             
.L_118b60:                      #        0x118b67  0      OPC=<label>         
  addl $0x1, %edx               #  247   0x118b67  3      OPC=addl_r32_imm8   
  nop                           #  248   0x118b6a  1      OPC=nop             
  nop                           #  249   0x118b6b  1      OPC=nop             
  nop                           #  250   0x118b6c  1      OPC=nop             
  nop                           #  251   0x118b6d  1      OPC=nop             
  nop                           #  252   0x118b6e  1      OPC=nop             
  nop                           #  253   0x118b6f  1      OPC=nop             
  nop                           #  254   0x118b70  1      OPC=nop             
  nop                           #  255   0x118b71  1      OPC=nop             
  nop                           #  256   0x118b72  1      OPC=nop             
  nop                           #  257   0x118b73  1      OPC=nop             
  nop                           #  258   0x118b74  1      OPC=nop             
  nop                           #  259   0x118b75  1      OPC=nop             
  nop                           #  260   0x118b76  1      OPC=nop             
  nop                           #  261   0x118b77  1      OPC=nop             
  nop                           #  262   0x118b78  1      OPC=nop             
  nop                           #  263   0x118b79  1      OPC=nop             
  nop                           #  264   0x118b7a  1      OPC=nop             
  nop                           #  265   0x118b7b  1      OPC=nop             
  nop                           #  266   0x118b7c  1      OPC=nop             
  nop                           #  267   0x118b7d  1      OPC=nop             
  nop                           #  268   0x118b7e  1      OPC=nop             
  nop                           #  269   0x118b7f  1      OPC=nop             
  nop                           #  270   0x118b80  1      OPC=nop             
  nop                           #  271   0x118b81  1      OPC=nop             
  nop                           #  272   0x118b82  1      OPC=nop             
  nop                           #  273   0x118b83  1      OPC=nop             
  nop                           #  274   0x118b84  1      OPC=nop             
  nop                           #  275   0x118b85  1      OPC=nop             
  nop                           #  276   0x118b86  1      OPC=nop             
.L_118b80:                      #        0x118b87  0      OPC=<label>         
  cmpl %esi, %edx               #  277   0x118b87  2      OPC=cmpl_r32_r32    
  jge .L_1189c0                 #  278   0x118b89  6      OPC=jge_label_1     
  movl %ecx, %eax               #  279   0x118b8f  2      OPC=movl_r32_r32    
  addl $0x38, %ecx              #  280   0x118b91  3      OPC=addl_r32_imm8   
  movl %eax, %eax               #  281   0x118b94  2      OPC=movl_r32_r32    
  cmpl $0x7, (%r15,%rax,1)      #  282   0x118b96  5      OPC=cmpl_m32_imm8   
  jne .L_118b60                 #  283   0x118b9b  2      OPC=jne_label       
  movl %eax, %eax               #  284   0x118b9d  2      OPC=movl_r32_r32    
  movq 0x30(%r15,%rax,1), %rdx  #  285   0x118b9f  5      OPC=movq_r64_m64    
  nop                           #  286   0x118ba4  1      OPC=nop             
  nop                           #  287   0x118ba5  1      OPC=nop             
  nop                           #  288   0x118ba6  1      OPC=nop             
  cmpw $0x3, 0xff07468(%rip)    #  289   0x118ba7  8      OPC=cmpw_m16_imm8   
  movl %edx, 0xff601be(%rip)    #  290   0x118baf  6      OPC=movl_m32_r32    
  movl %eax, %eax               #  291   0x118bb5  2      OPC=movl_r32_r32    
  movl 0x10(%r15,%rax,1), %edx  #  292   0x118bb7  5      OPC=movl_r32_m32    
  movl %edx, 0xff601a5(%rip)    #  293   0x118bbc  6      OPC=movl_m32_r32    
  nop                           #  294   0x118bc2  1      OPC=nop             
  nop                           #  295   0x118bc3  1      OPC=nop             
  nop                           #  296   0x118bc4  1      OPC=nop             
  nop                           #  297   0x118bc5  1      OPC=nop             
  nop                           #  298   0x118bc6  1      OPC=nop             
  jne .L_118be0                 #  299   0x118bc7  2      OPC=jne_label       
  addl $0x10020000, %edx        #  300   0x118bc9  6      OPC=addl_r32_imm32  
  movl %edx, 0xff60192(%rip)    #  301   0x118bcf  6      OPC=movl_m32_r32    
  nop                           #  302   0x118bd5  1      OPC=nop             
  nop                           #  303   0x118bd6  1      OPC=nop             
  nop                           #  304   0x118bd7  1      OPC=nop             
  nop                           #  305   0x118bd8  1      OPC=nop             
  nop                           #  306   0x118bd9  1      OPC=nop             
  nop                           #  307   0x118bda  1      OPC=nop             
  nop                           #  308   0x118bdb  1      OPC=nop             
  nop                           #  309   0x118bdc  1      OPC=nop             
  nop                           #  310   0x118bdd  1      OPC=nop             
  nop                           #  311   0x118bde  1      OPC=nop             
  nop                           #  312   0x118bdf  1      OPC=nop             
  nop                           #  313   0x118be0  1      OPC=nop             
  nop                           #  314   0x118be1  1      OPC=nop             
  nop                           #  315   0x118be2  1      OPC=nop             
  nop                           #  316   0x118be3  1      OPC=nop             
  nop                           #  317   0x118be4  1      OPC=nop             
  nop                           #  318   0x118be5  1      OPC=nop             
  nop                           #  319   0x118be6  1      OPC=nop             
.L_118be0:                      #        0x118be7  0      OPC=<label>         
  movl %eax, %eax               #  320   0x118be7  2      OPC=movl_r32_r32    
  movq 0x20(%r15,%rax,1), %rdx  #  321   0x118be9  5      OPC=movq_r64_m64    
  movl %eax, %eax               #  322   0x118bee  2      OPC=movl_r32_r32    
  movq 0x28(%r15,%rax,1), %rax  #  323   0x118bf0  5      OPC=movq_r64_m64    
  subl %edx, %eax               #  324   0x118bf5  2      OPC=subl_r32_r32    
  movl %edx, 0xff6016e(%rip)    #  325   0x118bf7  6      OPC=movl_m32_r32    
  movl %eax, 0xff6016c(%rip)    #  326   0x118bfd  6      OPC=movl_m32_r32    
  nop                           #  327   0x118c03  1      OPC=nop             
  nop                           #  328   0x118c04  1      OPC=nop             
  nop                           #  329   0x118c05  1      OPC=nop             
  nop                           #  330   0x118c06  1      OPC=nop             
  jmpq .L_118a00                #  331   0x118c07  5      OPC=jmpq_label_1    
  nop                           #  332   0x118c0c  1      OPC=nop             
  nop                           #  333   0x118c0d  1      OPC=nop             
  nop                           #  334   0x118c0e  1      OPC=nop             
  nop                           #  335   0x118c0f  1      OPC=nop             
  nop                           #  336   0x118c10  1      OPC=nop             
  nop                           #  337   0x118c11  1      OPC=nop             
  nop                           #  338   0x118c12  1      OPC=nop             
  nop                           #  339   0x118c13  1      OPC=nop             
  nop                           #  340   0x118c14  1      OPC=nop             
  nop                           #  341   0x118c15  1      OPC=nop             
  nop                           #  342   0x118c16  1      OPC=nop             
  nop                           #  343   0x118c17  1      OPC=nop             
  nop                           #  344   0x118c18  1      OPC=nop             
  nop                           #  345   0x118c19  1      OPC=nop             
  nop                           #  346   0x118c1a  1      OPC=nop             
  nop                           #  347   0x118c1b  1      OPC=nop             
  nop                           #  348   0x118c1c  1      OPC=nop             
  nop                           #  349   0x118c1d  1      OPC=nop             
  nop                           #  350   0x118c1e  1      OPC=nop             
  nop                           #  351   0x118c1f  1      OPC=nop             
  nop                           #  352   0x118c20  1      OPC=nop             
  nop                           #  353   0x118c21  1      OPC=nop             
  nop                           #  354   0x118c22  1      OPC=nop             
  nop                           #  355   0x118c23  1      OPC=nop             
  nop                           #  356   0x118c24  1      OPC=nop             
  nop                           #  357   0x118c25  1      OPC=nop             
  nop                           #  358   0x118c26  1      OPC=nop             
                                                                              
.size get_tls_info, .-get_tls_info

