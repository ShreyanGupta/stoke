  .text
  .globl _ZNK10__cxxabiv120__si_class_type_info12__do_dyncastEiNS_17__class_type_info10__sub_kindEPKS1_PKvS4_S6_RNS1_16__dyncast_resultE
  .type _ZNK10__cxxabiv120__si_class_type_info12__do_dyncastEiNS_17__class_type_info10__sub_kindEPKS1_PKvS4_S6_RNS1_16__dyncast_resultE, @function

#! file-offset 0x1220c0
#! rip-offset  0xe20c0
#! capacity    288 bytes

# Text                                                                                                                             #  Line  RIP      Bytes  Opcode              
._ZNK10__cxxabiv120__si_class_type_info12__do_dyncastEiNS_17__class_type_info10__sub_kindEPKS1_PKvS4_S6_RNS1_16__dyncast_resultE:  #        0xe20c0  0      OPC=<label>         
  movl %edi, %edi                                                                                                                  #  1     0xe20c0  2      OPC=movl_r32_r32    
  movl %ecx, %ecx                                                                                                                  #  2     0xe20c2  2      OPC=movl_r32_r32    
  movl %r8d, %r8d                                                                                                                  #  3     0xe20c4  3      OPC=movl_r32_r32    
  movl %edi, %edi                                                                                                                  #  4     0xe20c7  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rdi,1), %r11d                                                                                                     #  5     0xe20c9  5      OPC=movl_r32_m32    
  movl %r9d, %r9d                                                                                                                  #  6     0xe20ce  3      OPC=movl_r32_r32    
  movl 0x8(%rsp), %eax                                                                                                             #  7     0xe20d1  4      OPC=movl_r32_m32    
  movl %ecx, %ecx                                                                                                                  #  8     0xe20d5  2      OPC=movl_r32_r32    
  cmpl %r11d, 0x4(%r15,%rcx,1)                                                                                                     #  9     0xe20d7  5      OPC=cmpl_m32_r32    
  nop                                                                                                                              #  10    0xe20dc  1      OPC=nop             
  nop                                                                                                                              #  11    0xe20dd  1      OPC=nop             
  nop                                                                                                                              #  12    0xe20de  1      OPC=nop             
  nop                                                                                                                              #  13    0xe20df  1      OPC=nop             
  movl 0x10(%rsp), %r10d                                                                                                           #  14    0xe20e0  5      OPC=movl_r32_m32    
  je .L_e2140                                                                                                                      #  15    0xe20e5  2      OPC=je_label        
  cmpl %eax, %r8d                                                                                                                  #  16    0xe20e7  3      OPC=cmpl_r32_r32    
  je .L_e21a0                                                                                                                      #  17    0xe20ea  6      OPC=je_label_1      
  nop                                                                                                                              #  18    0xe20f0  1      OPC=nop             
  nop                                                                                                                              #  19    0xe20f1  1      OPC=nop             
  nop                                                                                                                              #  20    0xe20f2  1      OPC=nop             
  nop                                                                                                                              #  21    0xe20f3  1      OPC=nop             
  nop                                                                                                                              #  22    0xe20f4  1      OPC=nop             
  nop                                                                                                                              #  23    0xe20f5  1      OPC=nop             
  nop                                                                                                                              #  24    0xe20f6  1      OPC=nop             
  nop                                                                                                                              #  25    0xe20f7  1      OPC=nop             
  nop                                                                                                                              #  26    0xe20f8  1      OPC=nop             
  nop                                                                                                                              #  27    0xe20f9  1      OPC=nop             
  nop                                                                                                                              #  28    0xe20fa  1      OPC=nop             
  nop                                                                                                                              #  29    0xe20fb  1      OPC=nop             
  nop                                                                                                                              #  30    0xe20fc  1      OPC=nop             
  nop                                                                                                                              #  31    0xe20fd  1      OPC=nop             
  nop                                                                                                                              #  32    0xe20fe  1      OPC=nop             
  nop                                                                                                                              #  33    0xe20ff  1      OPC=nop             
.L_e2100:                                                                                                                          #        0xe2100  0      OPC=<label>         
  movl %edi, %edi                                                                                                                  #  34    0xe2100  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rdi,1), %edi                                                                                                      #  35    0xe2102  5      OPC=movl_r32_m32    
  movl %edi, %edi                                                                                                                  #  36    0xe2107  2      OPC=movl_r32_r32    
  movl (%r15,%rdi,1), %r11d                                                                                                        #  37    0xe2109  4      OPC=movl_r32_m32    
  movl %r10d, 0x10(%rsp)                                                                                                           #  38    0xe210d  5      OPC=movl_m32_r32    
  movl %eax, 0x8(%rsp)                                                                                                             #  39    0xe2112  4      OPC=movl_m32_r32    
  movl %r11d, %r11d                                                                                                                #  40    0xe2116  3      OPC=movl_r32_r32    
  movl 0x1c(%r15,%r11,1), %eax                                                                                                     #  41    0xe2119  5      OPC=movl_r32_m32    
  xchgw %ax, %ax                                                                                                                   #  42    0xe211e  2      OPC=xchgw_ax_r16    
  andl $0xffffffe0, %eax                                                                                                           #  43    0xe2120  6      OPC=andl_r32_imm32  
  nop                                                                                                                              #  44    0xe2126  1      OPC=nop             
  nop                                                                                                                              #  45    0xe2127  1      OPC=nop             
  nop                                                                                                                              #  46    0xe2128  1      OPC=nop             
  addq %r15, %rax                                                                                                                  #  47    0xe2129  3      OPC=addq_r64_r64    
  jmpq %rax                                                                                                                        #  48    0xe212c  2      OPC=jmpq_r64        
  nop                                                                                                                              #  49    0xe212e  1      OPC=nop             
  nop                                                                                                                              #  50    0xe212f  1      OPC=nop             
  nop                                                                                                                              #  51    0xe2130  1      OPC=nop             
  nop                                                                                                                              #  52    0xe2131  1      OPC=nop             
  nop                                                                                                                              #  53    0xe2132  1      OPC=nop             
  nop                                                                                                                              #  54    0xe2133  1      OPC=nop             
  nop                                                                                                                              #  55    0xe2134  1      OPC=nop             
  nop                                                                                                                              #  56    0xe2135  1      OPC=nop             
  nop                                                                                                                              #  57    0xe2136  1      OPC=nop             
  nop                                                                                                                              #  58    0xe2137  1      OPC=nop             
  nop                                                                                                                              #  59    0xe2138  1      OPC=nop             
  nop                                                                                                                              #  60    0xe2139  1      OPC=nop             
  nop                                                                                                                              #  61    0xe213a  1      OPC=nop             
  nop                                                                                                                              #  62    0xe213b  1      OPC=nop             
  nop                                                                                                                              #  63    0xe213c  1      OPC=nop             
  nop                                                                                                                              #  64    0xe213d  1      OPC=nop             
  nop                                                                                                                              #  65    0xe213e  1      OPC=nop             
  nop                                                                                                                              #  66    0xe213f  1      OPC=nop             
  nop                                                                                                                              #  67    0xe2140  1      OPC=nop             
  nop                                                                                                                              #  68    0xe2141  1      OPC=nop             
  nop                                                                                                                              #  69    0xe2142  1      OPC=nop             
  nop                                                                                                                              #  70    0xe2143  1      OPC=nop             
  nop                                                                                                                              #  71    0xe2144  1      OPC=nop             
  nop                                                                                                                              #  72    0xe2145  1      OPC=nop             
.L_e2140:                                                                                                                          #        0xe2146  0      OPC=<label>         
  testl %esi, %esi                                                                                                                 #  73    0xe2146  2      OPC=testl_r32_r32   
  movl %r10d, %r10d                                                                                                                #  74    0xe2148  3      OPC=movl_r32_r32    
  movl %r8d, (%r15,%r10,1)                                                                                                         #  75    0xe214b  4      OPC=movl_m32_r32    
  movl %r10d, %r10d                                                                                                                #  76    0xe214f  3      OPC=movl_r32_r32    
  movl %edx, 0x4(%r15,%r10,1)                                                                                                      #  77    0xe2152  5      OPC=movl_m32_r32    
  js .L_e21c0                                                                                                                      #  78    0xe2157  2      OPC=js_label        
  addl %r8d, %esi                                                                                                                  #  79    0xe2159  3      OPC=addl_r32_r32    
  cmpl %eax, %esi                                                                                                                  #  80    0xe215c  2      OPC=cmpl_r32_r32    
  sete %al                                                                                                                         #  81    0xe215e  3      OPC=sete_r8         
  movzbl %al, %eax                                                                                                                 #  82    0xe2161  3      OPC=movzbl_r32_r8   
  xchgw %ax, %ax                                                                                                                   #  83    0xe2164  2      OPC=xchgw_ax_r16    
  leal 0x1(%rax,%rax,4), %eax                                                                                                      #  84    0xe2166  4      OPC=leal_r32_m16    
  movl %r10d, %r10d                                                                                                                #  85    0xe216a  3      OPC=movl_r32_r32    
  movl %eax, 0xc(%r15,%r10,1)                                                                                                      #  86    0xe216d  5      OPC=movl_m32_r32    
  nop                                                                                                                              #  87    0xe2172  1      OPC=nop             
  nop                                                                                                                              #  88    0xe2173  1      OPC=nop             
  nop                                                                                                                              #  89    0xe2174  1      OPC=nop             
  nop                                                                                                                              #  90    0xe2175  1      OPC=nop             
  nop                                                                                                                              #  91    0xe2176  1      OPC=nop             
  nop                                                                                                                              #  92    0xe2177  1      OPC=nop             
  nop                                                                                                                              #  93    0xe2178  1      OPC=nop             
  nop                                                                                                                              #  94    0xe2179  1      OPC=nop             
  nop                                                                                                                              #  95    0xe217a  1      OPC=nop             
  nop                                                                                                                              #  96    0xe217b  1      OPC=nop             
  nop                                                                                                                              #  97    0xe217c  1      OPC=nop             
  nop                                                                                                                              #  98    0xe217d  1      OPC=nop             
  nop                                                                                                                              #  99    0xe217e  1      OPC=nop             
  nop                                                                                                                              #  100   0xe217f  1      OPC=nop             
  nop                                                                                                                              #  101   0xe2180  1      OPC=nop             
  nop                                                                                                                              #  102   0xe2181  1      OPC=nop             
  nop                                                                                                                              #  103   0xe2182  1      OPC=nop             
  nop                                                                                                                              #  104   0xe2183  1      OPC=nop             
  nop                                                                                                                              #  105   0xe2184  1      OPC=nop             
  nop                                                                                                                              #  106   0xe2185  1      OPC=nop             
.L_e2180:                                                                                                                          #        0xe2186  0      OPC=<label>         
  popq %r11                                                                                                                        #  107   0xe2186  2      OPC=popq_r64_1      
  xorl %eax, %eax                                                                                                                  #  108   0xe2188  2      OPC=xorl_r32_r32    
  andl $0xffffffe0, %r11d                                                                                                          #  109   0xe218a  7      OPC=andl_r32_imm32  
  nop                                                                                                                              #  110   0xe2191  1      OPC=nop             
  nop                                                                                                                              #  111   0xe2192  1      OPC=nop             
  nop                                                                                                                              #  112   0xe2193  1      OPC=nop             
  nop                                                                                                                              #  113   0xe2194  1      OPC=nop             
  addq %r15, %r11                                                                                                                  #  114   0xe2195  3      OPC=addq_r64_r64    
  jmpq %r11                                                                                                                        #  115   0xe2198  3      OPC=jmpq_r64        
  nop                                                                                                                              #  116   0xe219b  1      OPC=nop             
  nop                                                                                                                              #  117   0xe219c  1      OPC=nop             
  nop                                                                                                                              #  118   0xe219d  1      OPC=nop             
  nop                                                                                                                              #  119   0xe219e  1      OPC=nop             
  nop                                                                                                                              #  120   0xe219f  1      OPC=nop             
  nop                                                                                                                              #  121   0xe21a0  1      OPC=nop             
  nop                                                                                                                              #  122   0xe21a1  1      OPC=nop             
  nop                                                                                                                              #  123   0xe21a2  1      OPC=nop             
  nop                                                                                                                              #  124   0xe21a3  1      OPC=nop             
  nop                                                                                                                              #  125   0xe21a4  1      OPC=nop             
  nop                                                                                                                              #  126   0xe21a5  1      OPC=nop             
  nop                                                                                                                              #  127   0xe21a6  1      OPC=nop             
  nop                                                                                                                              #  128   0xe21a7  1      OPC=nop             
  nop                                                                                                                              #  129   0xe21a8  1      OPC=nop             
  nop                                                                                                                              #  130   0xe21a9  1      OPC=nop             
  nop                                                                                                                              #  131   0xe21aa  1      OPC=nop             
  nop                                                                                                                              #  132   0xe21ab  1      OPC=nop             
  nop                                                                                                                              #  133   0xe21ac  1      OPC=nop             
.L_e21a0:                                                                                                                          #        0xe21ad  0      OPC=<label>         
  movl %r9d, %r9d                                                                                                                  #  134   0xe21ad  3      OPC=movl_r32_r32    
  cmpl %r11d, 0x4(%r15,%r9,1)                                                                                                      #  135   0xe21b0  5      OPC=cmpl_m32_r32    
  jne .L_e2100                                                                                                                     #  136   0xe21b5  6      OPC=jne_label_1     
  movl %r10d, %r10d                                                                                                                #  137   0xe21bb  3      OPC=movl_r32_r32    
  movl %edx, 0x8(%r15,%r10,1)                                                                                                      #  138   0xe21be  5      OPC=movl_m32_r32    
  jmpq .L_e2180                                                                                                                    #  139   0xe21c3  2      OPC=jmpq_label      
  nop                                                                                                                              #  140   0xe21c5  1      OPC=nop             
  nop                                                                                                                              #  141   0xe21c6  1      OPC=nop             
  nop                                                                                                                              #  142   0xe21c7  1      OPC=nop             
  nop                                                                                                                              #  143   0xe21c8  1      OPC=nop             
  nop                                                                                                                              #  144   0xe21c9  1      OPC=nop             
  nop                                                                                                                              #  145   0xe21ca  1      OPC=nop             
  nop                                                                                                                              #  146   0xe21cb  1      OPC=nop             
  nop                                                                                                                              #  147   0xe21cc  1      OPC=nop             
.L_e21c0:                                                                                                                          #        0xe21cd  0      OPC=<label>         
  cmpl $0xfffffffe, %esi                                                                                                           #  148   0xe21cd  6      OPC=cmpl_r32_imm32  
  nop                                                                                                                              #  149   0xe21d3  1      OPC=nop             
  nop                                                                                                                              #  150   0xe21d4  1      OPC=nop             
  nop                                                                                                                              #  151   0xe21d5  1      OPC=nop             
  jne .L_e2180                                                                                                                     #  152   0xe21d6  2      OPC=jne_label       
  movl %r10d, %r10d                                                                                                                #  153   0xe21d8  3      OPC=movl_r32_r32    
  movl $0x1, 0xc(%r15,%r10,1)                                                                                                      #  154   0xe21db  9      OPC=movl_m32_imm32  
  jmpq .L_e2180                                                                                                                    #  155   0xe21e4  2      OPC=jmpq_label      
  nop                                                                                                                              #  156   0xe21e6  1      OPC=nop             
  nop                                                                                                                              #  157   0xe21e7  1      OPC=nop             
  nop                                                                                                                              #  158   0xe21e8  1      OPC=nop             
  nop                                                                                                                              #  159   0xe21e9  1      OPC=nop             
  nop                                                                                                                              #  160   0xe21ea  1      OPC=nop             
  nop                                                                                                                              #  161   0xe21eb  1      OPC=nop             
  nop                                                                                                                              #  162   0xe21ec  1      OPC=nop             
  nop                                                                                                                              #  163   0xe21ed  1      OPC=nop             
  nop                                                                                                                              #  164   0xe21ee  1      OPC=nop             
  nop                                                                                                                              #  165   0xe21ef  1      OPC=nop             
  nop                                                                                                                              #  166   0xe21f0  1      OPC=nop             
  nop                                                                                                                              #  167   0xe21f1  1      OPC=nop             
  nop                                                                                                                              #  168   0xe21f2  1      OPC=nop             
                                                                                                                                                                                
.size _ZNK10__cxxabiv120__si_class_type_info12__do_dyncastEiNS_17__class_type_info10__sub_kindEPKS1_PKvS4_S6_RNS1_16__dyncast_resultE, .-_ZNK10__cxxabiv120__si_class_type_info12__do_dyncastEiNS_17__class_type_info10__sub_kindEPKS1_PKvS4_S6_RNS1_16__dyncast_resultE

