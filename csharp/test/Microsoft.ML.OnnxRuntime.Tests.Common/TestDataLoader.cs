using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.NetworkInformation;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    internal static class TestDataLoader
    {
        internal static byte[] LoadModelFromEmbeddedResource(string path)
        {
            var assembly = typeof(TestDataLoader).Assembly;
            byte[] model = null;

            var resourceName = assembly.GetManifestResourceNames().Single(p => p.EndsWith("." + path));
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            {
                using (MemoryStream memoryStream = new MemoryStream())
                {
                    stream.CopyTo(memoryStream);
                    model = memoryStream.ToArray();
                }
            }

            return model;
        }


        internal static float[] LoadTensorFromEmbeddedResource(string path)
        {
            var tensorData = new List<float>();
            var assembly = typeof(TestDataLoader).Assembly;

            var resourceName = assembly.GetManifestResourceNames().Single(p => p.EndsWith("." + path));
            using (StreamReader inputFile = new StreamReader(assembly.GetManifestResourceStream(resourceName)))
            {
                inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        static NamedOnnxValue LoadTensorPb(Onnx.TensorProto tensor, string nodeName, NodeMetadata nodeMeta)
        {
            if (nodeMeta.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new InvalidDataException($"Metadata for: '{nodeName}' has a type: '{nodeMeta.OnnxValueType}'" +
                    $" but loading as tensor: '{tensor.Name}'");
            }

            var protoDt = (Tensors.TensorElementType)tensor.DataType;
            var metaElementType = nodeMeta.ElementDataType;
            if (!((protoDt == metaElementType) ||
                (protoDt == TensorElementType.UInt16 &&
                (metaElementType == TensorElementType.BFloat16 || metaElementType == TensorElementType.Float16))))
                throw new InvalidDataException($"{tensor.DataType} is expected to be equal to: {metaElementType}");

            // Tensors within Sequences may have no dimensions as the standard allows
            // different dimensions for each tensor element of the sequence
            if (nodeMeta.Dimensions.Length > 0 && nodeMeta.Dimensions.Length != tensor.Dims.Count)
            {
                throw new InvalidDataException($"node: '{nodeName}' nodeMeta.Dim.Length: {nodeMeta.Dimensions.Length} " +
                    $"is expected to be equal to tensor.Dims.Count {tensor.Dims.Count}");
            }

            var intDims = new int[tensor.Dims.Count];
            for (int i = 0; i < tensor.Dims.Count; i++)
            {
                intDims[i] = (int)tensor.Dims[i];
            }

            for (int i = 0; i < nodeMeta.Dimensions.Length; i++)
            {
                if ((nodeMeta.Dimensions[i] != -1) && (nodeMeta.Dimensions[i] != tensor.Dims[i]))
                    throw new InvalidDataException($"Node: '{nodeName}' dimension at idx {i} is {nodeMeta.Dimensions}[{i}] " +
                        $"is expected to either be -1 or {tensor.Dims[i]}");
            }

            // element type for Float16 and BFloat16 in the loaded tensor would always be uint16, so
            // we want to use element type from metadata
            return CreateNamedOnnxValueFromTensor(nodeName, tensor, metaElementType, intDims);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromTensor(string nodeName, Onnx.TensorProto tensor, TensorElementType elementType, int[] intDims)
        {
            switch (elementType)
            {
                case TensorElementType.Float:
                    return CreateNamedOnnxValueFromRawData<float>(nodeName, tensor.RawData.ToArray(), sizeof(float), intDims);
                case TensorElementType.Double:
                    return CreateNamedOnnxValueFromRawData<double>(nodeName, tensor.RawData.ToArray(), sizeof(double), intDims);
                case TensorElementType.Int32:
                    return CreateNamedOnnxValueFromRawData<int>(nodeName, tensor.RawData.ToArray(), sizeof(int), intDims);
                case TensorElementType.UInt32:
                    return CreateNamedOnnxValueFromRawData<uint>(nodeName, tensor.RawData.ToArray(), sizeof(uint), intDims);
                case TensorElementType.Int16:
                    return CreateNamedOnnxValueFromRawData<short>(nodeName, tensor.RawData.ToArray(), sizeof(short), intDims);
                case TensorElementType.UInt16:
                    return CreateNamedOnnxValueFromRawData<ushort>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
                case TensorElementType.Int64:
                    return CreateNamedOnnxValueFromRawData<long>(nodeName, tensor.RawData.ToArray(), sizeof(long), intDims);
                case TensorElementType.UInt64:
                    return CreateNamedOnnxValueFromRawData<ulong>(nodeName, tensor.RawData.ToArray(), sizeof(ulong), intDims);
                case TensorElementType.UInt8:
                    return CreateNamedOnnxValueFromRawData<byte>(nodeName, tensor.RawData.ToArray(), sizeof(byte), intDims);
                case TensorElementType.Int8:
                    return CreateNamedOnnxValueFromRawData<sbyte>(nodeName, tensor.RawData.ToArray(), sizeof(sbyte), intDims);
                case TensorElementType.Bool:
                    return CreateNamedOnnxValueFromRawData<bool>(nodeName, tensor.RawData.ToArray(), sizeof(bool), intDims);
                case TensorElementType.Float16:
                    return CreateNamedOnnxValueFromRawData<Float16>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
                case TensorElementType.BFloat16:
                    return CreateNamedOnnxValueFromRawData<BFloat16>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
                case TensorElementType.String:
                    return CreateNamedOnnxValueFromStringTensor(tensor, nodeName, intDims);
                default:
                    throw new InvalidDataException($"Tensors of type: " + elementType.ToString() + 
                        " not currently supported in the LoadTensorFromEmbeddedResource");
            }
        }

        internal static NamedOnnxValue LoadTensorFromEmbeddedResourcePb(string path, string nodeName, NodeMetadata nodeMeta)
        {
            Onnx.TensorProto tensor = null;

            var assembly = typeof(TestDataLoader).Assembly;

            using (Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.TestData.{path}"))
            {
                tensor = Onnx.TensorProto.Parser.ParseFrom(stream);
            }

            return LoadTensorPb(tensor, nodeName, nodeMeta);
        }

        private static string MakeSequenceElementName(string nodeName, string seqName, int seqNum)
        {
            if(seqName.Length > 0)
                return $"{nodeName}.{seqName}.seq.{seqNum}";
            else
                return $"{nodeName}._.seq.{seqNum}";
        }

        internal static NamedOnnxValue LoadOnnxValueFromFilePb(string fullFilename, string nodeName, NodeMetadata nodeMeta)
        {
            // No sparse tensor support yet
            //Set buffer size to 4MB
            int readBufferSize = 4194304;
            using (var file = new FileStream(fullFilename, FileMode.Open, FileAccess.Read, FileShare.Read, readBufferSize))
            {
                if (nodeMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                {
                    var tensor = Onnx.TensorProto.Parser.ParseFrom(file);
                    return LoadTensorPb(tensor, nodeName, nodeMeta);
                }

                if (nodeMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_SEQUENCE)
                {
                    var sequence = Onnx.SequenceProto.Parser.ParseFrom(file);
                    return CreateNamedOnnxValueFromSequence(sequence, nodeName, nodeMeta);
                }

                if (nodeMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_MAP)
                {
                    var map = Onnx.MapProto.Parser.ParseFrom(file);
                    //if (map.KeyType == (int)OnnxValueType.ONNX_TYPE_STRING &&
                    //    map.ValueType == (int)OnnxValueType.ONNX_TYPE_TENSOR)
                    //{
                    //    var elemMeta = nodeMeta.AsMapMetadata().ValueMeta;
                    //    var mapOfTensors = new Dictionary<string, NamedOnnxValue>(map.StringToTensor.Count);
                    //    foreach (var kv in map.StringToTensor)
                    //    {
                    //        var namedOnnxValue = LoadTensorPb(kv.Value, nodeName, elemMeta);
                    //        mapOfTensors.Add(kv.Key, namedOnnxValue);
                    //    }
                    //    return NamedOnnxValue.CreateFromDictionary(nodeName, mapOfTensors);
                    //}
                }
            }

            //else if (nodeMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_OPTIONAL)
            //{
            //    var containedTypeInfo = nodeMeta.AsOptionalMetadata().ElementMeta;
            //    return LoadOnnxValueFromFilePb(fullFilename, nodeName, containedTypeInfo);
            //}

            throw new ArgumentException($"Unable to load value type {nodeMeta.OnnxValueType} not implemented");
        }

        internal static void SequenceCheckMatchTensor(string nodeName, SequenceMetadata meta, Onnx.SequenceProto sequence)
        {
            if (meta.ElementMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR)
                return;

            throw new InvalidDataException($"Sequence node: '{nodeName}' " +
                $"has element type: '{(Onnx.SequenceProto.Types.DataType)sequence.ElemType}'" +
                $" expected: '{meta.ElementMeta.OnnxValueType}'");
        }

        internal static void SequenceCheckMatchSequence(string nodeName, SequenceMetadata meta, Onnx.SequenceProto sequence)
        {
            if (meta.ElementMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_SEQUENCE)
                return;

            throw new InvalidDataException($"Sequence node: '{nodeName}' " +
                $"has element type: '{(Onnx.SequenceProto.Types.DataType)sequence.ElemType}'" +
                $" expected: '{meta.ElementMeta.OnnxValueType}'");
        }

        internal static void SequenceCheckMatchMap(string nodeName, SequenceMetadata meta, Onnx.SequenceProto sequence)
        {
            if (meta.ElementMeta.OnnxValueType == OnnxValueType.ONNX_TYPE_MAP)
                return;

            throw new InvalidDataException($"Sequence node: '{nodeName}' " +
                $"has element type: '{(Onnx.SequenceProto.Types.DataType)sequence.ElemType}'" +
                $" expected: '{meta.ElementMeta.OnnxValueType}'");
        }


        internal static NamedOnnxValue CreateNamedOnnxValueFromSequence(Onnx.SequenceProto sequence, string nodeName, NodeMetadata nodeMeta)
        {
            var sequenceMeta = nodeMeta.AsSequenceMetadata();
            var elemMeta = sequenceMeta.ElementMeta;

            int seqNum = 0;
            var seqElemType = (Onnx.SequenceProto.Types.DataType)sequence.ElemType;
            switch (seqElemType)
            {
                case Onnx.SequenceProto.Types.DataType.Tensor:
                    {
                        SequenceCheckMatchTensor(nodeName, sequenceMeta, sequence);
                        var sequenceOfTensors = new List<NamedOnnxValue>(sequence.TensorValues.Count);
                        foreach (var tensor in sequence.TensorValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            var namedOnnxValue = LoadTensorPb(tensor, elemName, elemMeta);
                            sequenceOfTensors.Add(namedOnnxValue);
                        }
                        return NamedOnnxValue.CreateFromSequence(nodeName, sequenceOfTensors);
                    }
                case Onnx.SequenceProto.Types.DataType.Sequence:
                    {
                        SequenceCheckMatchSequence(nodeName, sequenceMeta, sequence);
                        var seqOfSequences = new List<NamedOnnxValue>(sequence.SequenceValues.Count);
                        foreach (var s in sequence.SequenceValues)
                        {
                            var elemName = MakeSequenceElementName(nodeName, sequence.Name, seqNum++);
                            seqOfSequences.Add(CreateNamedOnnxValueFromSequence(s, elemName, elemMeta));
                        }
                        return NamedOnnxValue.CreateFromSequence(nodeName, seqOfSequences);
                    }
                default:
                    throw new NotImplementedException($"Sequence loading does not support element type: " +
                        $"'{(Onnx.SequenceProto.Types.DataType)sequence.ElemType}'");
            }

        }

        //internal static CreateNamedOnnxValueFromMap(Onnx.MapProto map, string nodeName, NodeMetadata nodeMetadata)
        //{
        //    var mapMeta = nodeMetadata.AsMapMetadata();

        //    if (mapMeta.KeyDataType != (TensorElementType)map.KeyType)
        //    {
        //        throw new InvalidDataException($"Node: '{nodeName}' map key type expected: " +
        //                               $"'{mapMeta.KeyDataType}', loaded from test data: '{(TensorElementType)map.KeyType}'");
        //    }

        //    var valueMeta = mapMeta.ValueMetadata;
        //    //if(map.Values.ElemType != mapMeta.ValueMetadata.OnnxValueType)



        //}
        internal static NamedOnnxValue CreateNamedOnnxValueFromRawData<T>(string name, byte[] rawData, int elemWidth, int[] dimensions)
        {
            T[] typedArr = new T[rawData.Length / elemWidth];
            var typeOf = typeof(T);
            if (typeOf == typeof(Float16) || typeOf == typeof(BFloat16))
            {
                using (var memSrcHandle = new Memory<byte>(rawData).Pin())
                using (var memDstHandle = new Memory<T>(typedArr).Pin())
                {
                    unsafe
                    {
                        Buffer.MemoryCopy(memSrcHandle.Pointer, memDstHandle.Pointer, typedArr.Length * elemWidth, rawData.Length);
                    }
                }
            }
            else
            {
                Buffer.BlockCopy(rawData, 0, typedArr, 0, rawData.Length);
            }
            var dt = new DenseTensor<T>(typedArr, dimensions);
            return NamedOnnxValue.CreateFromTensor<T>(name, dt);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromStringTensor(Onnx.TensorProto tensor, string nodeName, int[] dimensions)
        {
            if (tensor.DataType != (int)Onnx.TensorProto.Types.DataType.String)
            {
                throw new ArgumentException("Expecting string data");
            }

            string[] strArray = new string[tensor.StringData.Count];
            for (int i = 0; i < tensor.StringData.Count; ++i)
            {
                strArray[i] = System.Text.Encoding.UTF8.GetString(tensor.StringData[i].ToByteArray());
            }

            var dt = new DenseTensor<string>(strArray, dimensions);
            return NamedOnnxValue.CreateFromTensor<string>(nodeName, dt);
        }

        internal static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }
    }
}